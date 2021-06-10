`timescale 1 ns / 1 ns

module RNN (input clk, input reset,
            input [31:0] writeW, input [31:0] writeH, input [31:0] writeU, input [31:0] writeX, input [31:0] writeV, input writeenable,
            input readport,
            output reg [31:0] Y);

    wire[31:0] mem2PU_W;
    wire[31:0] mem2PU_H;
    wire[31:0] mem2PU_U;
    wire[31:0] mem2PU_X;
    wire[31:0] mem2PU_V;

    wire[31:0] PU2mem_Y;
    wire[31:0] PU2mem_H;

    wire clk_enable;
    wire clk_enable_out;

    RAM mainmemory( .clk(clk), .reset(reset),
                    .writeeneable(write), .writeport(port),
                    .writeW(writeW), .writeH(writeH), .writeU(writeU), .writeX(writeX), .writeV(writeV),
                    .readport(readport),
                    .readW(mem2PU_W), .readH(mem2PU_H), .readU(mem2PU_U), .readX(mem2PU_X), .readV(mem2PU_V));

    RNN_Node PU1(   .clk(clk), .reset(reset), .clk_enable(clk_enable), 
                    .W(mem2PU_W), .h_t_1(mem2PU_H), .U(mem2PU_U), .x_t(mem2PU_X), .V(mem2PU_V), .ce_out(clk_enable_out), .y_t(PU2mem_Y), .h_t(PU2mem_H));

endmodule