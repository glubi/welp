#include <stdio.h>
#include <stdlib.h>

typedef struct no {
    int data;
    struct no * prox;
}no;

void inserir(no ** raiz, int v){
    no * novo = (no*)malloc(sizeof(no));
    novo->data=v;
    if((*raiz)==NULL){
        *raiz=novo;
        novo->prox=NULL;
    }else{
        novo->prox=*raiz;
        *raiz = novo;
    }
}

void remover(no ** raiz, int v){
    if((*raiz)==NULL){
        return;
    }
    no * aux;
    if((*raiz)->data==v){
        aux = (*raiz)->prox;
        free((*raiz));
        *raiz=aux;
    }else{
        aux = (*raiz);
        while (aux->prox!=NULL && aux->prox->data!=v) {
            aux = aux->prox;
        }
        if(aux->prox!=NULL && aux->prox->data==v){
            no * temp =  aux->prox->prox;
            free(aux->prox);
            aux->prox=temp;
        }
    }
}

int main(int argc, char const *argv[]) {
    no * raiz =  NULL;
    inserir(&raiz,1);
    inserir(&raiz,2);
    inserir(&raiz,3);
    inserir(&raiz,4);
    inserir(&raiz,5);
    inserir(&raiz,6);
    inserir(&raiz,7);


    remover(&raiz,1);
    remover(&raiz,2);
    remover(&raiz,3);
    remover(&raiz,4);
    remover(&raiz,5);
    remover(&raiz,6);
    remover(&raiz,7);

    printf("kdmdkmd");
    return 0;
}

