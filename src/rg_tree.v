/*
 * Building blocks for RiskGate-ML.
 *
 * rg_tree_step : one decision-tree comparison.  Same semantics as
 *                scikit-learn: go RIGHT if feature > threshold, else LEFT.
 *                The tree is walked one level per clock (3 clocks for depth 3),
 *                so the chip needs a single feature selector and comparator.
 *
 * rg_latch     : a W-bit transparent latch (sky130 dlxtp).  Used for the
 *                host-written configuration registers: a latch is about half
 *                the area of a flip-flop plus its hold-value multiplexer.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

`default_nettype none

module rg_tree_step (
    input  wire [47:0] feats,   // 6 features x 8 bits, feature k = feats[8k +: 8]
    input  wire [10:0] node,    // {feature[2:0], threshold[7:0]}
    output wire        right    // 1 = go right (feature > threshold)
);

  reg [7:0] f;
  always @(*) begin
    case (node[10:8])
      3'd0:    f = feats[7:0];
      3'd1:    f = feats[15:8];
      3'd2:    f = feats[23:16];
      3'd3:    f = feats[31:24];
      3'd4:    f = feats[39:32];
      3'd5:    f = feats[47:40];
      default: f = 8'd0;
    endcase
  end

  assign right = f > node[7:0];

endmodule


module rg_latch #(
    parameter W = 8
) (
    input  wire         en,
    input  wire [W-1:0] d,
    output reg  [W-1:0] q
);

  always @(*) begin
    if (en) q = d;
  end

endmodule
