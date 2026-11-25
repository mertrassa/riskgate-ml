/*
 * Depth-3 binary decision tree, evaluated along a single root-to-leaf path
 * (3 comparators instead of 7).  Same semantics as scikit-learn:
 *     go RIGHT if feature > threshold, else LEFT.
 * Node numbering is heap order: children of node n are 2n+1 and 2n+2.
 * Leaf index = {b0, b1, b2} where bk is the branch taken at depth k.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

`default_nettype none

module rg_tree (
    input  wire [47:0] feats,   // 6 features x 8 bits, feature k = feats[8k +: 8]
    input  wire [76:0] nodes,   // node n = nodes[11n +: 11] = {feature[2:0], thr[7:0]}
    input  wire  [7:0] leaves,  // label per leaf
    output wire  [2:0] leaf,
    output wire        label
);

  function [7:0] fsel;
    input [2:0]  s;
    input [47:0] f;
    begin
      case (s)
        3'd0:    fsel = f[7:0];
        3'd1:    fsel = f[15:8];
        3'd2:    fsel = f[23:16];
        3'd3:    fsel = f[31:24];
        3'd4:    fsel = f[39:32];
        3'd5:    fsel = f[47:40];
        default: fsel = 8'd0;
      endcase
    end
  endfunction

  // depth 0
  wire [10:0] n0 = nodes[10:0];
  wire        b0 = fsel(n0[10:8], feats) > n0[7:0];

  // depth 1: node 1 or 2
  wire [10:0] n1 = b0 ? nodes[2*11 +: 11] : nodes[1*11 +: 11];
  wire        b1 = fsel(n1[10:8], feats) > n1[7:0];

  // depth 2: node 3..6
  reg  [10:0] n2;
  always @(*) begin
    case ({b0, b1})
      2'b00: n2 = nodes[3*11 +: 11];
      2'b01: n2 = nodes[4*11 +: 11];
      2'b10: n2 = nodes[5*11 +: 11];
      default: n2 = nodes[6*11 +: 11];
    endcase
  end
  wire b2 = fsel(n2[10:8], feats) > n2[7:0];

  assign leaf  = {b0, b1, b2};
  assign label = leaves[leaf];

endmodule
