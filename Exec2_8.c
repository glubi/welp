#include <stdio.h>
#include <stdlib.h>

typedef struct no {
    int data;
    struct no * prox;
}no;

typedef struct Lista{
    no * start;
    no * end;
}Lista;

int inserir(Lista * l, int valor){
    no * novo;
    novo = (no*)malloc(sizeof(no));
    novo->data =  valor;
    novo->prox = novo;
    if(l->start==NULL){
        l->start = novo;
        l->end = novo;
    }else{
        no * aux = l->start;
        while (aux->data!=valor && aux->prox!=l->start) {
            aux = aux->prox;
        }
        if(aux->data==valor){
            return 1;//valor ja existe
        }
        l->end->prox = novo;
        l->end = novo;
        l->end->prox = l->start;
    }
    return 0;
}


void remover(Lista * l, int valor){
    no * aux  =  l->start;
    if(l->start->data==valor){
        if(l->start->prox==l->start){
            l->start=NULL;
            l->end=NULL;
        }else{
            l->end->prox=l->start->prox;
            l->start=l->start->prox;
        }
        free(aux);
    }else{
        no * ant =  l->start;
        no * aux = aux->prox;
        while (aux->data!=valor && aux->prox!=l->start) {
            ant = aux;
            aux = aux->prox;
        }
        if(aux->data==valor){
            ant->prox=aux->prox;
            if(aux==l->end){
                l->end=aux;
            }
            free(aux);
        }
    }
}




int main(int argc, char const *argv[]) {
    Lista l;
    l.end=NULL;
    l.start=NULL;
    inserir(&l,10);
    inserir(&l,10);
    inserir(&l,20);
    inserir(&l,30);
    inserir(&l,40);
    inserir(&l,50);
    inserir(&l,15);
    inserir(&l,35);
    inserir(&l,2);
    inserir(&l,4);
    inserir(&l,10);
    inserir(&l,56);
    inserir(&l,1);

    remover(&l,10);
    remover(&l,10);
    remover(&l,10);
    remover(&l,20);
    remover(&l,30);
    remover(&l,40);
    remover(&l,50);
    remover(&l,15);
    remover(&l,35);
    remover(&l,2);
    remover(&l,4);
    remover(&l,10);
    remover(&l,56);
    remover(&l,1);

    printf("kdsmsks");
    return 0;
}

