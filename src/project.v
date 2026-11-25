/*
 * RiskGate-ML : pre-trade risk gateway with an on-chip decision-tree
 *               anomaly detector, for Tiny Tapeout.
 *
 * Every order is checked by five hard rules (kill switch, fat-finger price
 * band, order size, position limit, order-rate throttle) and scored by a
 * depth-3 decision tree trained offline in Python (see ../ml).  The tree is
 * walked one level per clock; the decision is committed 4 clocks (160 ns at
 * 25 MHz) after the SUBMIT write.
 * The ML layer can only ADD rejections - it can never approve an order that
 * breaks a hard rule.
 *
 * Copyright (c) 2026  -  SPDX-License-Identifier: Apache-2.0
 */

`default_nettype none

module tt_um_riskgate_ml (
    input  wire [7:0] ui_in,    // DATA bus (write data)
    output wire [7:0] uo_out,   // read data, or 7-segment display
    input  wire [7:0] uio_in,   // [3:0] ADDR, [4] WR strobe, [5] KILL
    output wire [7:0] uio_out,  // [6] ACCEPT LED, [7] REJECT LED
    output wire [7:0] uio_oe,
    input  wire       ena,
    input  wire       clk,
    input  wire       rst_n
);

  // ------------------------------------------------------------------
  // Register map (write).  Reads use the same ADDR pins, see R_* below.
  // ------------------------------------------------------------------
  localparam [3:0] A_CTRL     = 4'h0;  // [0]disp7seg [1]ml_en [2]ml_enforce [3]kill [4]manual_time
  localparam [3:0] A_PRICE    = 4'h1;  // staged order price (ticks)
  localparam [3:0] A_QTY      = 4'h2;  // staged order quantity
  localparam [3:0] A_SUBMIT   = 4'h3;  // [0]side (0=buy 1=sell) -> evaluate order
  localparam [3:0] A_REF      = 4'h4;  // reference price
  localparam [3:0] A_BAND     = 4'h5;  // max |price-ref|
  localparam [3:0] A_MAXQTY   = 4'h6;  // max order quantity
  localparam [3:0] A_POSLIM   = 4'h7;  // max |net position|, in units of 8 shares
  localparam [3:0] A_RATEMAX  = 4'h8;  // max order attempts per window
  localparam [3:0] A_RATEWIN  = 4'h9;  // window length = RATEWIN+1 ticks
  localparam [3:0] A_TIMEBASE = 4'hA;  // [3:0] tick = 2^n clock cycles
  localparam [3:0] A_NODESEL  = 4'hB;  // [6:4] node index, [2:0] feature select
  localparam [3:0] A_NODETHR  = 4'hC;  // threshold for the node last selected
  localparam [3:0] A_LEAVES   = 4'hD;  // bit i = label of leaf i (1 = anomalous)
  localparam [3:0] A_SIDE     = 4'hE;  // [0] side used for feature preview
  localparam [3:0] A_CMD      = 4'hF;  // [0]clr pos [1]clr window+streak [2]step tick [3]clr status

  localparam [3:0] R_STATUS   = 4'h0;
  localparam [3:0] R_POSLO    = 4'h1;
  localparam [3:0] R_POSHI    = 4'h2;
  localparam [3:0] R_F0       = 4'h3;  // |price - ref|
  localparam [3:0] R_F1       = 4'h4;  // qty
  localparam [3:0] R_F2       = 4'h5;  // |position after order| / 8 (saturating)
  localparam [3:0] R_F3       = 4'h6;  // order attempts in current window
  localparam [3:0] R_F4       = 4'h7;  // ticks since previous order (saturating)
  localparam [3:0] R_LEAF     = 4'h8;  // tree leaf hit by the last order
  localparam [3:0] R_WINTICKS = 4'h9;
  localparam [3:0] R_F5       = 4'hA;  // consecutive same-side orders incl. this one
  localparam [3:0] R_ID       = 4'hF;

  wire [7:0] data = ui_in;
  wire [3:0] addr = uio_in[3:0];

  // ------------------------------------------------------------------
  // Asynchronous host interface: synchronise WR and KILL, detect WR edge.
  // ADDR/DATA must be held stable by the host while WR is high.
  // ------------------------------------------------------------------
  reg [2:0] wr_sync;
  reg [1:0] kill_sync;
  always @(posedge clk) begin
    if (!rst_n) begin
      wr_sync   <= 3'b000;
      kill_sync <= 2'b00;
    end else begin
      wr_sync   <= {wr_sync[1:0], uio_in[4]};
      kill_sync <= {kill_sync[0], uio_in[5]};
    end
  end
  wire wr        = wr_sync[1] & ~wr_sync[2];
  wire we_submit = wr && (addr == A_SUBMIT);
  wire we_cmd    = wr && (addr == A_CMD);

  // ------------------------------------------------------------------
  // Configuration / staging registers, held in latches.
  // A latch is open while its register is written (the one-clock WR pulse,
  // during which the host holds ADDR/DATA stable) and while in reset, when it
  // loads its reset value.  rst_hold switches the data back to the bus one
  // clock AFTER reset releases, so the latches close on the reset value.
  // ------------------------------------------------------------------
  reg rst_hold;
  always @(posedge clk) rst_hold <= ~rst_n;

  wire        in_rst = ~rst_n;
  wire [15:0] wsel   = wr ? (16'd1 << addr) : 16'd0;
  wire  [7:0] d_00   = rst_hold ? 8'd0   : data;
  wire  [7:0] d_ff   = rst_hold ? 8'd255 : data;
  wire  [7:0] d_100  = rst_hold ? 8'd100 : data;
  wire  [3:0] d_10   = rst_hold ? 4'd10  : data[3:0];

  wire [4:0] ctrl;
  wire [7:0] price, qty, ref_price, band, max_qty, pos_lim, rate_max, rate_win;
  wire [3:0] tb_shift;
  wire [2:0] node_ptr;
  wire [7:0] leaves;
  wire       side_r;

  rg_latch #(5) l_ctrl  (.en(in_rst | wsel[A_CTRL]),     .d(d_00[4:0]), .q(ctrl));
  rg_latch #(8) l_price (.en(in_rst | wsel[A_PRICE]),    .d(d_100),     .q(price));
  rg_latch #(8) l_qty   (.en(in_rst | wsel[A_QTY]),      .d(d_00),      .q(qty));
  rg_latch #(1) l_side  (.en(in_rst | wsel[A_SUBMIT] | wsel[A_SIDE]), .d(d_00[0]), .q(side_r));
  rg_latch #(8) l_ref   (.en(in_rst | wsel[A_REF]),      .d(d_100),     .q(ref_price));
  rg_latch #(8) l_band  (.en(in_rst | wsel[A_BAND]),     .d(d_ff),      .q(band));
  rg_latch #(8) l_maxq  (.en(in_rst | wsel[A_MAXQTY]),   .d(d_ff),      .q(max_qty));
  rg_latch #(8) l_posl  (.en(in_rst | wsel[A_POSLIM]),   .d(d_ff),      .q(pos_lim));
  rg_latch #(8) l_rmax  (.en(in_rst | wsel[A_RATEMAX]),  .d(d_ff),      .q(rate_max));
  rg_latch #(8) l_rwin  (.en(in_rst | wsel[A_RATEWIN]),  .d(d_ff),      .q(rate_win));
  rg_latch #(4) l_tb    (.en(in_rst | wsel[A_TIMEBASE]), .d(d_10)     , .q(tb_shift));
  rg_latch #(3) l_nptr  (.en(in_rst | wsel[A_NODESEL]),  .d(d_00[6:4]), .q(node_ptr));
  rg_latch #(8) l_leaf  (.en(in_rst | wsel[A_LEAVES]),   .d(d_00),      .q(leaves));

  // Decision-tree nodes: node i = nodes[11*i +: 11] = {feature[2:0], threshold[7:0]}
  // Reset value (feature 0, threshold 255) always branches left.
  wire [76:0] nodes;
  genvar gi;
  generate
    for (gi = 0; gi < 7; gi = gi + 1) begin : g_node
      rg_latch #(3) l_feat (.en(in_rst | (wsel[A_NODESEL] & (data[6:4] == gi))),
                            .d(d_00[2:0]), .q(nodes[11*gi+8 +: 3]));
      rg_latch #(8) l_thr  (.en(in_rst | (wsel[A_NODETHR] & (node_ptr == gi))),
                            .d(d_ff),      .q(nodes[11*gi +: 8]));
    end
  endgenerate

  // ------------------------------------------------------------------
  // Time base: auto ticks every 2^tb_shift cycles, or manual ticks via CMD
  // (manual mode gives cycle-exact, deterministic market replay).
  // ------------------------------------------------------------------
  // tick_auto = carry into bit tb_shift of the prescaler, i.e. the low
  // tb_shift bits are all ones (re-uses the incrementer's carry chain).
  reg  [14:0] presc;
  wire [15:0] carry = {&presc[14:0], &presc[13:0], &presc[12:0], &presc[11:0],
                       &presc[10:0], &presc[9:0],  &presc[8:0],  &presc[7:0],
                       &presc[6:0],  &presc[5:0],  &presc[4:0],  &presc[3:0],
                       &presc[2:0],  &presc[1:0],  presc[0],     1'b1};
  wire        tick_auto = carry[tb_shift];
  wire        tick_raw  = (tick_auto & ~ctrl[4]) | (we_cmd & data[2]);
  // While the tree is walked (3 clocks) the features must not move: a tick
  // arriving then is held in tick_pend and applied right after.
  wire        frozen;
  reg         tick_pend;
  wire        tick      = (tick_raw | tick_pend) & ~frozen;
  wire        clr_pos   = we_cmd & data[0];
  wire        clr_win   = we_cmd & data[1];
  wire        clr_stat  = we_cmd & data[3];

  // ------------------------------------------------------------------
  // Running state
  // ------------------------------------------------------------------
  reg [11:0] pos;        // signed net position (shares)
  reg  [7:0] win_ticks;  // ticks elapsed in current rate window
  reg  [7:0] win_count;  // order attempts in current rate window
  reg  [7:0] idle;       // ticks since previous order, saturating
  reg  [7:0] streak;     // consecutive same-side orders (0 = no order yet)
  reg        last_side;
  reg  [7:0] status;
  reg  [2:0] leaf_r;
  reg        decided;

  // ------------------------------------------------------------------
  // Feature extraction (combinational, for the staged order)
  // ------------------------------------------------------------------
  wire        side_eff = side_r;
  wire  [8:0] pdiff = {1'b0, price} - {1'b0, ref_price};
  wire  [7:0] f0    = pdiff[8] ? (8'd0 - pdiff[7:0]) : pdiff[7:0];
  wire  [7:0] f1 = qty;

  wire [12:0] pos_ext   = {pos[11], pos};
  wire [12:0] qty_ext   = {5'd0, qty};
  wire [12:0] pos_after = side_eff ? (pos_ext - qty_ext) : (pos_ext + qty_ext);
  wire [12:0] pos_abs   = pos_after[12] ? (13'd0 - pos_after) : pos_after;
  wire  [7:0] f2 = (pos_abs[12:11] != 2'b00) ? 8'd255 : pos_abs[10:3];
  wire  [7:0] f3 = win_count;
  wire  [7:0] f4 = idle;
  // f5: same-side streak including this order (runaway algos hammer one side)
  wire  [7:0] f5 = (streak != 8'd0 && side_eff == last_side)
                   ? ((streak == 8'd255) ? 8'd255 : streak + 8'd1)
                   : 8'd1;

  // ------------------------------------------------------------------
  // Hard rules
  // ------------------------------------------------------------------
  wire kill_fail  = kill_sync[1] | ctrl[3];
  wire price_fail = (f0 > band);
  wire size_fail  = (qty == 8'd0) || (qty > max_qty);
  wire pos_fail   = (pos_abs > {2'b00, pos_lim, 3'b000});
  wire rate_fail  = (win_count >= rate_max);
  wire rules_fail = kill_fail | price_fail | size_fail | pos_fail | rate_fail;

  // ------------------------------------------------------------------
  // ML layer: depth-3 decision tree, walked one level per clock.
  // ev[0..2]: compare at depth 0..2; ev[3]: commit the decision.
  // ------------------------------------------------------------------
  reg  [3:0] ev;
  reg  [2:0] idx;    // heap index of the node being compared
  reg  [2:0] path;   // branch bits so far = leaf index when done
  assign frozen = |ev[2:0];
  wire       commit = ev[3];

  reg [10:0] nsel;
  always @(*) begin
    case (idx)
      3'd0:    nsel = nodes[0*11 +: 11];
      3'd1:    nsel = nodes[1*11 +: 11];
      3'd2:    nsel = nodes[2*11 +: 11];
      3'd3:    nsel = nodes[3*11 +: 11];
      3'd4:    nsel = nodes[4*11 +: 11];
      3'd5:    nsel = nodes[5*11 +: 11];
      default: nsel = nodes[6*11 +: 11];
    endcase
  end

  wire go_right;
  rg_tree_step u_tree (
      .feats  ({f5, f4, f3, f2, f1, f0}),
      .node   (nsel),
      .right  (go_right)
  );

  always @(posedge clk) begin
    if (!rst_n) begin
      ev        <= 4'd0;
      idx       <= 3'd0;
      path      <= 3'd0;
      tick_pend <= 1'b0;
    end else begin
      ev        <= {ev[2:0], we_submit};
      tick_pend <= frozen & (tick_raw | tick_pend);
      if (we_submit)
        idx <= 3'd0;
      else if (frozen) begin
        idx  <= {idx[1:0], 1'b0} + 3'd1 + {2'b00, go_right};
        path <= {path[1:0], go_right};
      end
    end
  end

  wire [2:0] leaf       = path;
  wire       leaf_label = leaves[path];
  wire ml_hit     = ctrl[1] & leaf_label;   // reported even in shadow mode
  wire reject     = rules_fail | (ml_hit & ctrl[2]);
  wire accept_now = commit & ~reject;

  wire win_wrap = tick && (win_ticks == rate_win);

  always @(posedge clk) begin
    if (!rst_n) begin
      presc     <= 15'd0;
      idle      <= 8'd255;
      streak    <= 8'd0;
      last_side <= 1'b0;
      win_ticks <= 8'd0;
      win_count <= 8'd0;
      pos       <= 12'd0;
      status    <= 8'd0;
      leaf_r    <= 3'd0;
      decided   <= 1'b0;
    end else begin
      presc <= presc + 15'd1;

      if (commit)
        idle <= 8'd0;
      else if (tick && idle != 8'd255)
        idle <= idle + 8'd1;

      if (clr_win)
        streak <= 8'd0;
      else if (commit) begin
        streak    <= f5;
        last_side <= side_eff;
      end

      if (clr_win) begin
        win_ticks <= 8'd0;
        win_count <= 8'd0;
      end else begin
        if (tick)
          win_ticks <= win_wrap ? 8'd0 : win_ticks + 8'd1;
        // counts every order attempt (message-rate throttle), saturating
        if (win_wrap)
          win_count <= commit ? 8'd1 : 8'd0;
        else if (commit && win_count != 8'd255)
          win_count <= win_count + 8'd1;
      end

      if (clr_pos)
        pos <= 12'd0;
      else if (accept_now)
        pos <= pos_after[11:0];

      if (clr_stat) begin
        decided <= 1'b0;
        status  <= 8'd0;
        leaf_r  <= 3'd0;
      end else if (commit) begin
        decided <= 1'b1;
        status  <= {~status[7], ~reject, ml_hit, rate_fail,
                    pos_fail, size_fail, price_fail, kill_fail};
        leaf_r  <= leaf;
      end
    end
  end

  // ------------------------------------------------------------------
  // Outputs
  // ------------------------------------------------------------------
  reg [7:0] rdata;
  always @(*) begin
    case (addr)
      R_STATUS:   rdata = status;
      R_POSLO:    rdata = pos[7:0];
      R_POSHI:    rdata = {{4{pos[11]}}, pos[11:8]};
      R_F0:       rdata = f0;
      R_F1:       rdata = f1;
      R_F2:       rdata = f2;
      R_F3:       rdata = f3;
      R_F4:       rdata = f4;
      R_LEAF:     rdata = {5'd0, leaf_r};
      R_WINTICKS: rdata = win_ticks;
      R_F5:       rdata = f5;
      R_ID:       rdata = 8'h52;  // 'R'
      default:    rdata = 8'h00;
    endcase
  end

  // 7-segment (gfedcba) shows the deciding check; DP = ML flagged.
  reg [6:0] seg;
  always @(*) begin
    if (!decided)        seg = 7'h40;  // -
    else if (status[6])  seg = 7'h77;  // A  accepted
    else if (status[0])  seg = 7'h76;  // H  halted (kill switch)
    else if (status[1])  seg = 7'h73;  // P  price band
    else if (status[2])  seg = 7'h6D;  // 5  size
    else if (status[3])  seg = 7'h38;  // L  position limit
    else if (status[4])  seg = 7'h50;  // r  rate limit
    else                 seg = 7'h5E;  // d  ML detector
  end

  assign uo_out  = ctrl[0] ? {status[5], seg} : rdata;
  assign uio_out = {decided & ~status[6], decided & status[6], 6'b000000};
  assign uio_oe  = 8'b1100_0000;

  wire _unused = &{ena, uio_in[7:6], 1'b0};

endmodule
