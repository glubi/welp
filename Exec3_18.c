#include<stdio.h>
#include<stdlib.h>



typedef struct no{
    int info;
    struct no * esq;
    struct no * dir;

}no;

typedef struct pilha {
    no * ndata;
    struct pilha * prox;
}pilha;

void empilha(pilha ** cabeca, no * v){
    pilha * novo = (pilha*)malloc(sizeof(pilha));
    novo->ndata=v;
    novo->prox=(*cabeca);
    (*cabeca)=novo;
}


no * desempilha(pilha ** cabeca){
    if((*cabeca)==NULL){
        return NULL;
    }

    no * result = (*cabeca)->ndata;
    no * aux = (*cabeca);
    *cabeca=(*cabeca)->prox;
    free(aux);
    return result;
}

percurso_pre_ordem(no ** raiz){
    if((*raiz)==NULL)
        return;
    pilha * p = NULL;
    empilha(&p,(*raiz));

    do{
        no * aux = desempilha(&p);
        printf("%d, ",aux->info);
        if(aux->dir!=NULL){
            empilha(&p, aux->dir);
        }
        if(aux->esq!=NULL){
            empilha(&p, aux->esq);
        }
    }while (p!=NULL);
}

percurso_ordem(no ** raiz){
    if((*raiz)==NULL)
        return;

    pilha * p = NULL;
    no * aux = (*raiz);

    do{
        if(aux!=NULL){
            empilha(&p,aux);
            aux = aux->esq;
        }else{
            aux = desempilha(&p);
            printf("%d, ",aux->info);
            aux =  aux->dir;
        }
    }while (p!=NULL || aux!=NULL);
}


percurso_pos_ordem(no ** raiz){
    if((*raiz)==NULL)
        return;
    pilha * p = NULL;
    no * aux = (*raiz);
    no * q = (*raiz);

    do{
        while(aux->esq!=NULL){
            empilha(&p,aux);
            aux = aux->esq;
        }
        while (aux!=NULL && (aux->dir == NULL || aux->dir==q )) {
            printf("%d, ",aux->info);
            q = aux;
            if(p==NULL){
                return;
            }
            aux = desempilha(&p);
        }
        empilha(&p,aux);
        aux = aux->dir;
    }while (p!=NULL);
}




no * buscaBin(no ** raiz,int v){
    no * aux = (*raiz);
    while (1) {
        if(aux->info>v){
            if(aux->esq!=NULL){
                aux=aux->esq;
            }else{
                return aux;
            }
        }else if(aux->info<v){
            if(aux->dir!=NULL){
                aux=aux->dir;
            }else{
                return aux;
            }
        }else{
            return aux;
        }
    }
    return NULL;
}

void inserir(no ** raiz,int v){
    no * novo = (no*)malloc(sizeof(no));
    novo->dir=NULL;
    novo->esq=NULL;
    novo->info=v;

    if((*raiz)==NULL){
        *raiz=novo;
    }else{
        no * aux = buscaBin(&(*raiz),v);
        if(aux->info!=v){
            if(aux->info>v){
                aux->esq=novo;
            }else{
                aux->dir=novo;
            }
        }
    }

}

int main(){
    no * raiz =NULL;
    inserir(&raiz,6);
    inserir(&raiz,3);
    inserir(&raiz,1);
    inserir(&raiz,2);
    inserir(&raiz,8);
    inserir(&raiz,7);
    inserir(&raiz,9);
    percurso_pre_ordem(&raiz);
    printf("\n");
    percurso_ordem(&raiz);
    printf("\n");
    percurso_pos_ordem(&raiz);
    return 0;
}
