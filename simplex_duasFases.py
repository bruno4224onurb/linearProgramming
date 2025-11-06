import numpy as np

def simplex(c, A, b, indice, tol=1e-8):
    TOL = tol
    m,n = A.shape
    indice = list(indice)              # garantir lista mutável
    B = A[:, indice]
    N = [j for j in range(n) if j not in indice]
    iteracao = 1
    while True:
        # passo 1
        B = A[:, indice]
        N = [j for j in range(n) if j not in indice]
        print("-------------- Matriz Básica  --------------\n",A[:,indice])
        print("------------ Matriz Não Básica  ------------\n",A[:,N])
        print(f"--------------  Iteração: {iteracao}   --------------")
        try:
            xB = np.linalg.solve(B, b)
        except np.linalg.LinAlgError:
            print("Base singular — falha numérica.")
            return None, None, None
        # passo 2.1
        cB = c[indice]
        try:
            lambda_ = np.linalg.solve(B.T, cB)
        except np.linalg.LinAlgError:
            print("Falha ao resolver B^T λ = cB.")
            return None, None, None
        # passo 2.2: Custos relativos
        cN = c[N]
        AN = A[:, N]
        rN = cN - lambda_.T @ AN
        # passo 2.3
        j_min_idx = np.argmin(rN)
        c_rel_min = rN[j_min_idx]
        j_in = N[j_min_idx]
        # passo 3
        if c_rel_min >= -TOL:
            x = np.zeros(n)
            x[indice] = xB
            val = c @ x

            print("\n!!!!!!!!  Solução ótima encontrada  !!!!!!!!\n")
            print(f"   Valor ótimo (função objetivo): {val:.6f}")
            print(f"   Vetor solução x*: {x}\n")

            return val, indice, x

        # passo 4
        a_k = A[:, j_in]
        try:
            y = np.linalg.solve(B, a_k)
        except np.linalg.LinAlgError:
            print("Falha ao resolver By = a_k.")
            return None, None, None

        # passo 5
        if np.all(y <= TOL):
            print("Problema ilimitado (todos elementos de y ≤ 0).")
            return None, None, None

        ratios = np.full(m, np.inf)
        for i in range(m):
            if y[i] > TOL:
                ratios[i] = xB[i] / y[i]

        i_out = np.argmin(ratios)

        # passo 6
        indice[i_out] = j_in
        iteracao += 1
        x = np.zeros(n)
        x[indice] = xB
        val = c @ x
        print(f"   Valor função objetivo: {val:.6f}")
        print(f"   Vetor candidato a solução: {x}")
        # O loop continua até encontrar ótimo ou detectar ilimitado


def primfase(c, A, b):
    m,n = A.shape
    # copiar A e b para não modificar fora da função
    A = A.copy()
    b = b.copy()

    for i in range(m):
        if b[i] < 0:
            A[i, :] *= -1
            b[i] *= -1

    identidade = np.eye(m)
    indice = []
    # teste de trivialidade: procurar colunas que sejam colunas da identidade
    for j in range(n):
        for k in range(m):
            if np.allclose(A[:, j], identidade[:, k]):
                # evitar duplicatas: só adiciona se aquela posição k ainda não mapeada
                if k not in [ (np.where(np.allclose(A[:, col], identidade[:, k]))[0] if False else None) ]:
                    indice.append(j)
                else:
                    # mesmo que não controlemos k diretamente, apenas adicionamos coluna j
                    # se ela representar alguma coluna padrão ainda não usada
                    indice.append(j)
                break
        if len(indice) == m:
            break

    # Observação: a verificação acima procura colunas idênticas a alguma coluna da identidade.
    # Se já encontrarmos m colunas independentes que formem a base, não precisamos da Fase I.
    if len(indice) == m:
        print("\n========= Fase I não é necessária  =========")
        print("====== problema com partição trivial  ======")
        print("-------------- Matriz Básica  --------------\n",A[:,indice])
        N = [j for j in range(n) if j not in indice]
        print("------------ Matriz Não Básica  ------------\n",A[:,N])
        return indice

    # problema auxiliar (adiciona variáveis artificiais)
    A_aux = np.hstack((A, np.eye(m)))
    c_aux = np.hstack((np.zeros(n), np.ones(m)))

    # CORREÇÃO: criar lista com índices das artificiais como base inicial
    indice_aux = list(range(n, n + m))

    print("\n== INICIANDO FASE I (problema auxiliar) ==")
    otimo, base_final, x = simplex(c_aux, A_aux, b, indice_aux)

    TOL = 1e-8
    if otimo is None:
        print("Fase I falhou.")
        return None

    if otimo > TOL:
        print("Problema auxiliar com F.O. ótima diferente de 0.")
        print("Problema original inviável.")
        return None
    else:
        print("Problema auxiliar com F.O. ótima igual a 0")
        print("Problema original viável (Fase I bem-sucedida)")
        # remover variáveis artificiais da base final (se houver)
        base_viavel = [j for j in base_final if j < n]
        # se a base viável tiver menos que m colunas (pode acontecer), tentamos completar:
        if len(base_viavel) < m:
            # procurar colunas independentes das originais para completar a base
            for j in range(n):
                if j not in base_viavel:
                    candidate = base_viavel + [j]
                    if len(candidate) <= m:
                        Bcand = A[:, candidate]
                        if np.linalg.matrix_rank(Bcand) == len(candidate):
                            base_viavel = candidate.copy()
                    if len(base_viavel) == m:
                        break
        return base_viavel


def ler_dados(caminho_arquivo):
    with open(caminho_arquivo, "r") as f:
        linhas = [linha.strip() for linha in f if linha.strip() and not linha.startswith("#")]
    m, n = map(int, linhas[0].split())
    A = np.array([list(map(float, linha.split())) for linha in linhas[1:1+m]])
    b = np.array(list(map(float, linhas[1+m].split())))
    c = np.array(list(map(float, linhas[2+m].split())))
    return c, A, b


# MAIN
if __name__ == "__main__":
    caminho = "ilimitada.txt"   # <- arquivo de entrada
    c, A, b = ler_dados(caminho)

    print("\n======  INICIANDO SIMPLEX DUAS FASES  ======")
    print("---------------  Matriz A:   ---------------\n", A)
    print("---------------   Vetor b:   ---------------\n", b)
    print("---------------   Vetor c:   ---------------\n", c)

    base_inicial = primfase(c, A, b)
    
    if base_inicial is not None:
        print("\n============  INICIANDO FASE II ============")
        simplex(c, A, b, base_inicial)
