`timescale 1 ns / 1 ns

module RNN (input clk, input reset,
            input writeenable, input [8:0] writeport,
            input [31:0] writeW, input [31:0] writeH, input [31:0] writeU, input [31:0] writeX, input [31:0] writeV, 
            input [8:0] readport,
            output reg [31:0] Y,
            input start, input [8:0] numlayers, output reg done);

    wire[31:0] mem2PU_W;
    wire[31:0] mem2PU_H;
    wire[31:0] mem2PU_U;
    wire[31:0] mem2PU_X;
    wire[31:0] mem2PU_V;

    wire[31:0] PU2mem_Y;
    wire[31:0] PU2mem_H;

    reg enablePU1;
    wire clk_enable_out;

    reg [8:0] counter;

    integer i;

    RAM mainmemory( .clk(clk), .reset(reset),
                    .writeenable(writeenable), .writeport(writeport),
                    .writeW(writeW), .writeH(writeH), .writeU(writeU), .writeX(writeX), .writeV(writeV),
                    .readport(readport),
                    .readW(mem2PU_W), .readH(mem2PU_H), .readU(mem2PU_U), .readX(mem2PU_X), .readV(mem2PU_V));

    RNN_Node PU1(   .clk(clk), .reset(reset), .clk_enable(enablePU1), 
                    .W(mem2PU_W), .h_t_1(mem2PU_H), .U(mem2PU_U), .x_t(mem2PU_X), .V(mem2PU_V), .ce_out(clk_enable_out), .y_t(PU2mem_Y), .h_t(PU2mem_H));

    //Timing logic
    always @ (posedge clk) begin
        if(reset) begin
            counter <= 9'b0;
            done <= 1'b0;
        end
        else if(start && counter == 9'b0) begin
            enablePU1 <= 1'b1;
            done <= 1'b0;
            counter <= counter + 1'b1;
        end
        else if(counter > 0 && counter < numlayers) begin
            counter <= counter + 1'b1;
        end
        else if (counter == numlayers) begin
            done <= 1'b1;
            Y <= PU2mem_Y;
            counter <= 0;
        end
        else begin
            done <= 1'b1;
        end
    end

endmodule