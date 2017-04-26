def qsort(lista):
    if len(lista) <= 1:
        return lista
    else:
        menor = []
        igual = []
        maior = []
        pivo = lista[0]
        for i in lista:
            if i < pivo:
                menor.append(i)
            elif i == pivo:
                igual.append(i)
            elif i > pivo:
                maior.append(i)
        return qsort(menor) + igual + qsort(maior)
      
 
def msort(lista):
    result = []
    if len(lista) <= 1:
        return lista
    else:
        mid = int(len(lista)/2)
        y = msort(lista[:mid])
        z = msort(lista[mid:])
        i = 0
        j = 0
    while i < len(y) and j < len(z):
        if y[i] > z[j]:
            result.append(z[j])
            j += 1
        else:
            result.append(y[i])
            i += 1
    result += y[i:]
    result += z[j:]
    return result
  
  
def bsearch(lista, x, fim=0, inicio=None):
    if inicio is None:
        inicio = len(lista)
    while fim < inicio:
        mid = (fim + inicio) // 2
        midvalue = lista[mid]
        if midvalue < x:
            fim = mid + 1
        elif midvalue > x:
            inicio = mid
        else:
            return (mid, len(lista))
    return -1
