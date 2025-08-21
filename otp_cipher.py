class OneTimePad:
    @staticmethod
    def encrypt(message: str, key: str) -> str:
        """Cifra mensagem usando OTP"""
        if len(key) < len(message):
            raise ValueError("Chave deve ser pelo menos do tamanho da mensagem")
        
        # Converter mensagem para binário
        message_bits = ''.join(format(ord(char), '08b') for char in message)
        
        # XOR com a chave
        encrypted_bits = ''
        for i, bit in enumerate(message_bits):
            encrypted_bits += str(int(bit) ^ int(key[i]))
        
        return encrypted_bits
    
    @staticmethod
    def decrypt(encrypted_bits: str, key: str) -> str:
        """Decifra mensagem usando OTP"""
        # XOR com a chave (operação inversa)
        decrypted_bits = ''
        for i, bit in enumerate(encrypted_bits):
            decrypted_bits += str(int(bit) ^ int(key[i]))
        
        # Converter de volta para texto
        message = ''
        for i in range(0, len(decrypted_bits), 8):
            byte = decrypted_bits[i:i+8]
            if len(byte) == 8:
                message += chr(int(byte, 2))
        
        return message