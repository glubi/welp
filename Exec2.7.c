#include <stdio.h>
#define  MAX 4

int esq = -1; int dir = MAX;
int pilha[MAX];



void init(){
        esq = MAX; dir = -1;
        for (size_t i = 0; i < MAX; i++) {
                pilha[i]=0;
        }
}

void printAll(){
        for (int i = 0; i < MAX; i++) {
                printf("%d,", pilha[i]);
        }
        printf("\n");
}


int empilha(int pNum, int v){
        if(dir-1==esq) {
                return 1; //over flow
        }
        int index = pNum==1 ? ++esq : --dir;
        pilha[index]=v;
        return 0;
}

int desempilha(int pNum){
        int index;
        if(pNum == 1 ) {
                if(esq<0) return -1;
                else{
                        index = pilha[esq--];
                        pilha[esq+1]=-1;
                }
        }else{
                if(dir>=MAX) return -1;
                else {
                        index = pilha[dir++];
                        pilha[dir-1]=-1;
                }
        }
        return index;
}



int main(int argc, char const *argv[]) {
        empilha(1,1);
        empilha(1,2);
        empilha(1,3);
        empilha(1,4);
        desempilha(1);
        desempilha(1);
        desempilha(1);
        desempilha(1);
        desempilha(1);
        empilha(1,3);
        empilha(2,3);
        printAll();
        return 0;
}
