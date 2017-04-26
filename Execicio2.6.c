#include <stdio.h>

#define  MAX 4

int esq = MAX; int dir = -1;

int deque[MAX];

int insert(int lado, int v){
    int index;
    if(lado==1)  { //dir
        index = dir+1 % MAX;
        if(index !=esq ) {
            deque[index]=v;
            dir = index;
        }else{
            return 1;
        }
    }else{ //esq
        index  =  esq-1%MAX;
        if(index !=dir ) {
            deque[index]=v;
            esq = index;
        }else{
            return 1;
        }
    }
    return 0;
}

void init(){
    esq = MAX; dir = -1;
    for (size_t i = 0; i < MAX; i++) {
        deque[i]=0;
    }
}

void printAll(){
    for (size_t i = 0; i < MAX; i++) {
        printf("%d", deque[i]);
    }
    printf("\n");
}


int remover(int lado){
    int index;
    int result;

    if(dir==esq){
        init();
    }
    if(lado==1)  { //dir
        if(dir == -1 && esq!=MAX){
            dir = MAX-1;
            result = deque[dir];
            deque[dir]=-1;
            dir--;
            return result;
        }
        if(dir>=0 && dir<MAX ) {
            result = deque[dir];
            deque[dir]=-1;
            if(dir==0){
                dir = MAX-1;
            }else{
                dir = dir - 1 % MAX;
            }
        }
    }else{
        if(esq>=0 &&  esq<MAX) {
            result = deque[esq];
            deque[esq] = -1;
            index = esq + 1 % MAX;
            esq = index;
        }
    }
    return result;
}


int main(int argc, char const *argv[]) {    
    init();
    insert(1,4);
    insert(1,1);
    insert(2,3);
    insert(2,2);

    remover(1);
    remover(1);
    remover(1);
    remover(1);
    printAll();

    remover(2);
    remover(2);
    remover(2);
    remover(2);

    insert(2,1);
    insert(2,3);
    insert(2,4);
    insert(2,5);

    remover(1);
    remover(1);
    remover(1);

    printAll();

    //remover(2);
    ///printf("%d\n",insert(2,10));
    ///printAll();
    return 0;
}
