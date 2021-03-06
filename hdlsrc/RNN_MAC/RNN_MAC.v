// -------------------------------------------------------------
// 
// File Name: hdlsrc\RNN_MAC\RNN_MAC.v
// Created: 2021-06-11 12:37:08
// 
// Generated by MATLAB 9.9 and HDL Coder 3.17
// 
// 
// -- -------------------------------------------------------------
// -- Rate and Clocking Details
// -- -------------------------------------------------------------
// Model base rate: 0.2
// Target subsystem base rate: 0.2
// 
// 
// Clock Enable  Sample Time
// -- -------------------------------------------------------------
// ce_out        0.2
// -- -------------------------------------------------------------
// 
// 
// Output Signal                 Clock Enable  Sample Time
// -- -------------------------------------------------------------
// h_t_1_W                       ce_out        0.2
// x_t_U                         ce_out        0.2
// -- -------------------------------------------------------------
// 
// -------------------------------------------------------------


// -------------------------------------------------------------
// 
// Module: RNN_MAC
// Source Path: RNN_MAC
// Hierarchy Level: 0
// 
// -------------------------------------------------------------

`timescale 1 ns / 1 ns

module RNN_MAC
          (clk,
           reset,
           clk_enable,
           W,
           h_t_1,
           U,
           x_t,
           ce_out,
           h_t_1_W,
           x_t_U);


  input   clk;
  input   reset;
  input   clk_enable;
  input   [31:0] W;  // single
  input   [31:0] h_t_1;  // single
  input   [31:0] U;  // single
  input   [31:0] x_t;  // single
  output  ce_out;
  output  [31:0] h_t_1_W;  // single
  output  [31:0] x_t_U;  // single


  wire enb;
  reg [31:0] Delay_out1;  // ufix32
  reg [31:0] Delay1_out1;  // ufix32
  wire [31:0] mulOutput;  // ufix32
  wire [31:0] Multiply_Add_out1;  // ufix32
  reg [31:0] Delay2_out1;  // ufix32
  reg [31:0] Delay5_out1;  // ufix32
  reg [31:0] Delay3_out1;  // ufix32
  wire [31:0] mulOutput_1;  // ufix32
  wire [31:0] Multiply_Add1_out1;  // ufix32
  reg [31:0] Delay4_out1;  // ufix32


  assign enb = clk_enable;

  always @(posedge clk or posedge reset)
    begin : Delay_process
      if (reset == 1'b1) begin
        Delay_out1 <= 32'h00000000;
      end
      else begin
        if (enb) begin
          Delay_out1 <= W;
        end
      end
    end



  always @(posedge clk or posedge reset)
    begin : Delay1_process
      if (reset == 1'b1) begin
        Delay1_out1 <= 32'h00000000;
      end
      else begin
        if (enb) begin
          Delay1_out1 <= h_t_1;
        end
      end
    end



  nfp_mul_single u_nfp_mul_comp (.nfp_in1(Delay_out1),  // single
                                 .nfp_in2(Delay1_out1),  // single
                                 .nfp_out(mulOutput)  // single
                                 );

  always @(posedge clk or posedge reset)
    begin : Delay2_process
      if (reset == 1'b1) begin
        Delay2_out1 <= 32'h00000000;
      end
      else begin
        if (enb) begin
          Delay2_out1 <= Multiply_Add_out1;
        end
      end
    end



  nfp_add_single u_nfp_add_comp (.nfp_in1(Delay2_out1),  // single
                                 .nfp_in2(mulOutput),  // single
                                 .nfp_out(Multiply_Add_out1)  // single
                                 );

  always @(posedge clk or posedge reset)
    begin : Delay5_process
      if (reset == 1'b1) begin
        Delay5_out1 <= 32'h00000000;
      end
      else begin
        if (enb) begin
          Delay5_out1 <= U;
        end
      end
    end



  always @(posedge clk or posedge reset)
    begin : Delay3_process
      if (reset == 1'b1) begin
        Delay3_out1 <= 32'h00000000;
      end
      else begin
        if (enb) begin
          Delay3_out1 <= x_t;
        end
      end
    end



  nfp_mul_single u_nfp_mul_comp_1 (.nfp_in1(Delay5_out1),  // single
                                   .nfp_in2(Delay3_out1),  // single
                                   .nfp_out(mulOutput_1)  // single
                                   );

  always @(posedge clk or posedge reset)
    begin : Delay4_process
      if (reset == 1'b1) begin
        Delay4_out1 <= 32'h00000000;
      end
      else begin
        if (enb) begin
          Delay4_out1 <= Multiply_Add1_out1;
        end
      end
    end



  nfp_add_single u_nfp_add_comp_1 (.nfp_in1(Delay4_out1),  // single
                                   .nfp_in2(mulOutput_1),  // single
                                   .nfp_out(Multiply_Add1_out1)  // single
                                   );

  assign ce_out = clk_enable;

  assign h_t_1_W = Multiply_Add_out1;

  assign x_t_U = Multiply_Add1_out1;

endmodule  // RNN_MAC

