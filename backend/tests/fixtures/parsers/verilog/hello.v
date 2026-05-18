module greeter (input wire clk, output reg out);
    always @(posedge clk) begin
        out <= 1'b1;
        format_signal();
    end
endmodule

task format_signal;
    begin
        $display("hi");
    end
endtask
