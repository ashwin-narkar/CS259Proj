`timescale 1 ns / 1 ns

module RAM( input clk, input reset, //control signals
            input writeenable, input[8:0] writeport,
            input[31:0] writeW, input[31:0] writeH, input[31:0] writeU, input[31:0] writeX, input[31:0] writeV,
            input[8:0] readport,
            output reg [31:0] readW, output reg [31:0] readH, output reg [31:0] readU, output reg [31:0] readX, output reg [31:0] readV);

    parameter RAMSIZE = 512;

    integer i;

    reg [31:0] Wreg [0:RAMSIZE-1];
    reg [31:0] Hreg [0:RAMSIZE-1];
    reg [31:0] Ureg [0:RAMSIZE-1];
    reg [31:0] Xreg [0:RAMSIZE-1];
    reg [31:0] Vreg [0:RAMSIZE-1];

    always @ (posedge clk) begin
        if(reset) begin
            for(i = 0; i < RAMSIZE; i = i + 1) begin
                Wreg[i] = 32'b0;
                Hreg[i] = 32'b0;
                Ureg[i] = 32'b0;
                Xreg[i] = 32'b0;
                Vreg[i] = 32'b0;
            end
        end
        else if (writeenable)begin
            Wreg[writeport] = writeW;
            Hreg[writeport] = writeH;
            Ureg[writeport] = writeU;
            Xreg[writeport] = writeX;
            Vreg[writeport] = writeV;
        end
        readW = Wreg[readport];
        readH = Hreg[readport];
        readU = Ureg[readport];
        readX = Xreg[readport];
        readV = Vreg[readport];
    end

endmodule