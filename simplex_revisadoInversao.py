import numpy as np

def simplex(c, A, b, indices_basicos):
    
    # obter dimensões (m restrições e n variáveis)
    m, n = A.shape

    # tolerância para comparações de ponto flutuante
    TOL = 1e-9

    # obter matriz basica B inicial
    B = A[:, indices_basicos]

    # inverter B apenas uma vez
    try:
        B_inv = np.linalg.inv(B)
    except np.linalg.LinAlgError:
        return {'status': 'erro', 'message': 'Matriz básica singular.'}

    # looping de iterações
    max_iter = 100 # número máximo de iterações para previnir looping infinito
    for iteracao in range(max_iter):
        print(f"--- Iteração {iteracao + 1} ---")
        print(f"Base atual: {[(i+1) for i in indices_basicos]}")

        # achar a partição não-básica de A (A = [B N])
        indices_nbasicos = [i for i in range(n) if i not in indices_basicos]

        N = A[:, indices_nbasicos]
        c_B = c[indices_basicos]
        c_N = c[indices_nbasicos]
        
        # --- Passo 1 ---
        # calcular solução básica atual
        x_B = B_inv @ b
        # valor da função atual
        valor_atual = c_B.T @ x_B
        print(f"Valor atual da função objetivo: {valor_atual:.4f}")

        # --- Passo 2 ---
        # 2.1) calcular vetor multiplicador simplex
        lamb_T = c_B.T @ B_inv
        # 2.2) custos relativos
        custos_relativos = c_N.T - lamb_T @ N
        # 2.3) escolher variável a entrar na base
        index_nbasico = np.argmin(custos_relativos) # (Regra de Dantzig)
        index_basico = indices_nbasicos[index_nbasico]
        print(f"Variável entrando na base: x{index_basico + 1} (custo relativo: {custos_relativos[index_nbasico]:.4f})")

        # --- Passo 3 ---
        # teste de otimalidade: se todos os custos relativos são >= 0, a solução é ótima
        if np.all(custos_relativos >= -TOL):
            x_final = np.zeros(n)
            x_final[indices_basicos] = x_B
            f_otimo = c.T @ x_final

            print("\nCondição de otimalidade atingida.")
            return {
                'status': 'otima',
                'solucao': x_final,
                'f': f_otimo,
                'base_final': [(i+1) for i in indices_basicos]
            }
        
        # --- Passo 4 ---
        # calcular a direção simplex
        a_Nk = A[:, index_basico]
        y = B_inv @ a_Nk

        # --- Passo 5 ---
        # determinar passo e variável a sair da base
        if (np.all(y <= TOL)): # se todos os elementos de y são <= 0, o problema é ilimitado
            return {'status': 'ilimitado', 'message': 'Problema ilimitado.'}
        
        # determinar a variável a sair da base
        menor_razao = float('inf')
        index_sai_da_base = -1

        for i in range(m):
            if y[i] > TOL:
                razao = x_B[i] / y[i]
                if razao < menor_razao:
                    menor_razao = razao
                    index_sai_da_base = i

        variavel_sai_da_base = indices_basicos[index_sai_da_base]
        print(f"Variável saindo da base: x{variavel_sai_da_base + 1} (razão: {menor_razao:.4f})")

        # --- Passo 6 ---
        # atualizar a inversa
        p = index_sai_da_base # índice da linha do pivô
        
        # pegar o valor do pivô
        pivo_val = y[p]
        
        # copiar a linha do pivô da B_inv antiga
        linha_pivo_ant = B_inv[p, :].copy()
        
        # dividir a linha do pivô pelo elemento pivô
        B_inv[p, :] = linha_pivo_ant / pivo_val
        
        # atualizar todas as outras linhas i != p
        # nova_linha_i = velha_linha_i - d[i] * nova_linha_p
        for i in range(m):
            if i != p:
                d_i = y[i] # elemento da coluna 'd' na linha 'i'
                B_inv[i, :] = B_inv[i, :] - d_i * B_inv[p, :]

        # --- Passo 7 ---
        # atualizar a base
        indices_basicos[index_sai_da_base] = index_basico
        print("-" * 40 + "\n")

    return {'status': 'excesso_iteracoes', 'message': 'Número máximo de iterações atingido.'}