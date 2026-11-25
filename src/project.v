/*
 * RiskGate-ML : pre-trade risk gateway with an on-chip decision-tree
 *               anomaly detector, for Tiny Tapeout.
 *
 * Every order is checked by five hard rules (kill switch, fat-finger price
 * band, order size, position limit, order-rate throttle) and, in parallel,
 * scored by a depth-3 decision tree trained offline in Python (see ../ml).
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
  // Configuration / staging registers
  // ------------------------------------------------------------------
  reg [4:0] ctrl;
  reg [7:0] price, qty, ref_price, band, max_qty, pos_lim, rate_max, rate_win;
  reg [3:0] tb_shift;
  reg [2:0] node_ptr;
  reg [7:0] leaves;
  reg       side_r;

  always @(posedge clk) begin
    if (!rst_n) begin
      ctrl      <= 5'b0;
      price     <= 8'd100;
      qty       <= 8'd0;
      ref_price <= 8'd100;
      band      <= 8'd255;
      max_qty   <= 8'd255;
      pos_lim   <= 8'd255;
      rate_max  <= 8'd255;
      rate_win  <= 8'd255;
      tb_shift  <= 4'd10;
      node_ptr  <= 3'd0;
      leaves    <= 8'd0;
      side_r    <= 1'b0;
    end else if (wr) begin
      case (addr)
        A_CTRL:     ctrl      <= data[4:0];
        A_PRICE:    price     <= data;
        A_QTY:      qty       <= data;
        A_SUBMIT:   side_r    <= data[0];
        A_REF:      ref_price <= data;
        A_BAND:     band      <= data;
        A_MAXQTY:   max_qty   <= data;
        A_POSLIM:   pos_lim   <= data;
        A_RATEMAX:  rate_max  <= data;
        A_RATEWIN:  rate_win  <= data;
        A_TIMEBASE: tb_shift  <= data[3:0];
        A_NODESEL:  node_ptr  <= data[6:4];
        A_LEAVES:   leaves    <= data;
        A_SIDE:     side_r    <= data[0];
        default: ;
      endcase
    end
  end

  // Decision-tree nodes: node i = nodes[11*i +: 11] = {feature[2:0], threshold[7:0]}
  // Reset value (feature 0, threshold 255) always branches left.
  reg [76:0] nodes;
  integer i;
  always @(posedge clk) begin
    if (!rst_n) begin
      for (i = 0; i < 7; i = i + 1)
        nodes[11*i +: 11] <= {3'd0, 8'd255};
    end else begin
      for (i = 0; i < 7; i = i + 1) begin
        if (wr && addr == A_NODESEL && data[6:4] == i)
          nodes[11*i+8 +: 3] <= data[2:0];
        if (wr && addr == A_NODETHR && node_ptr == i)
          nodes[11*i +: 8] <= data;
      end
    end
  end

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
  wire        tick      = (tick_auto & ~ctrl[4]) | (we_cmd & data[2]);
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
  wire        side_eff = we_submit ? data[0] : side_r;
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
  // ML layer: depth-3 decision tree
  // ------------------------------------------------------------------
  wire [2:0] leaf;
  wire       leaf_label;
  rg_tree u_tree (
      .feats  ({f5, f4, f3, f2, f1, f0}),
      .nodes  (nodes),
      .leaves (leaves),
      .leaf   (leaf),
      .label  (leaf_label)
  );
  wire ml_hit     = ctrl[1] & leaf_label;   // reported even in shadow mode
  wire reject     = rules_fail | (ml_hit & ctrl[2]);
  wire accept_now = we_submit & ~reject;

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

      if (we_submit)
        idle <= 8'd0;
      else if (tick && idle != 8'd255)
        idle <= idle + 8'd1;

      if (clr_win)
        streak <= 8'd0;
      else if (we_submit) begin
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
          win_count <= we_submit ? 8'd1 : 8'd0;
        else if (we_submit && win_count != 8'd255)
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
      end else if (we_submit) begin
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
