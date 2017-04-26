#include <stdio.h>
#include <stdlib.h>

#define maxTam 5
int inicio = 0;
int fim = 0;


int fila[maxTam];

int inserir(int v){
        if (fim+1%maxTam!=inicio) {
          fila[fim]=v;
          fim = fim+1%maxTam;
          return 0;
        }else{
          return 1;
        }
}

int main(int argc, char const *argv[]) {
        
        return 0;
}
