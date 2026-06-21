module uart_test_top (
    input wire CLOCK_50,       // Relógio de 50 MHz (PIN_Y2)
    input wire [0:0] KEY,      // Botão 0 para Reset (PIN_M23)
    input wire UART_RXD,       // Pino do cabo serial (PIN_AH23)
    
    output wire [7:0] LEDG,    // 8 LEDs Verdes (Mostrar o Byte)
    output reg [0:0] LEDR      // 1 LED Vermelho (Mostrar que validou)
);

    wire [7:0] fio_dado;
    wire fio_valido;

    // Instancia o SEU módulo já testado
    uart_rx #(
        .CLK_FREQ(50000000),
        .BAUD_RATE(115200)
    ) receptor (
        .clk(CLOCK_50),
        // Na placa DE2-115, os botões dão '0' quando apertados, 
        // então precisamos inverter (~) para o nosso 'rst' que é ativo em '1'
        .rst(~KEY[0]), 
        .rx_pin(UART_RXD),
        .data_out(fio_dado),
        .data_valid(fio_valido)
    );

    // 1. Liga o dado recebido direto nos 8 LEDs Verdes
    assign LEDG = fio_dado;

    // 2. Truque visual para o LED Vermelho
    // Como o 'fio_valido' dura só 20 nanosegundos (impossível de ver a olho nu),
    // nós fazemos o LED Vermelho "mudar de estado" toda vez que um pacote chega.
    always @(posedge CLOCK_50) begin
        if (~KEY[0]) begin
            LEDR[0] <= 1'b0;
        end else if (fio_valido) begin
            LEDR[0] <= ~LEDR[0]; // Inverte o estado (Acende/Apaga)
        end
    end

endmodule