#include <stdio.h>
#include <stdlib.h>


typedef struct no{
    struct no * ant;
    struct no * prox;
    int   data;
}no;

void insert(no ** raiz,int value){
    no * novo;
    novo = (no*)malloc(sizeof(no));
    novo->data=value;
    novo->prox=NULL;

    if(*raiz == NULL){
        novo->ant =NULL;
        *raiz=novo;
    }else{
        no * aux = (*raiz);
        while (aux->prox!=NULL) {
            aux = aux->prox;
        }
        novo->ant= aux;
        aux->prox=novo;
    }
}

void remover(no ** raiz, int v){
    if(*raiz==NULL)
        return;
    no * aux = (*raiz)->prox;
    if((*raiz)->data==v){
        *raiz=aux;
        if(aux!=NULL)
            aux->ant=NULL;
        free((*raiz));
    }else{
        while (aux!=NULL && aux->data!=v) {
            aux = aux->prox;
        }
        if(aux!=NULL){
            if(aux->prox!=NULL){
                aux->ant->prox=aux->prox;
                aux->prox->ant=aux->ant;
            }else{
                aux->ant->prox=NULL;

            }
            free(aux);
        }else{
            printf("não encontrado \n");
        }
    }
}



int main()
{
    no * raiz =NULL;
    insert(&raiz,1);
    insert(&raiz,2);
    insert(&raiz,3);
    insert(&raiz,333);
    insert(&raiz,5);
    insert(&raiz,6);
    insert(&raiz,7);
    insert(&raiz,8);
    insert(&raiz,9);

    remover(&raiz,5);
    remover(&raiz,1);
    remover(&raiz,7);
    remover(&raiz,2);
    remover(&raiz,9);
    remover(&raiz,3);
    remover(&raiz,6);
    remover(&raiz,4);
    remover(&raiz,8);




    printf("kdkdmdk");

    return 0;
}
