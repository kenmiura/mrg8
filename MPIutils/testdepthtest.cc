#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "Timer.hh"

#ifdef T3E
typedef short int32;
#else
typedef int int32;
#endif

void main(int argc,char *argv[]){
  register int i,n;
  int bufsize=512*512; /* 32Meg buffer */
  float *fdepthbuf = new float[bufsize];
  int32 *idepthbuf = new int32[bufsize];
  int32 *pixelbuf = new int32[bufsize];
  Timer t;

  memset(pixelbuf,0,bufsize*sizeof(int32));
  memset(fdepthbuf,0,bufsize*sizeof(float));
  memset(idepthbuf,0,bufsize*sizeof(int32));

  t.reset();
  t.start();
  for(n=0;n<1000;n++){
	register float d=(float)n;
    for(i=0;i<bufsize;i++){
	if(fdepthbuf[i]<=d){ 
	  (pixelbuf[i])+=1;
	  fdepthbuf[i]=d;
	}
    }
  }
  t.stop();
  t.print("fdepth: ");

  t.reset();
  t.start();
  for(n=0;n<1000;n++) {
	register float d=(float)n;
    for(i=0;i<bufsize;i++){
	register int id=(int)(d+i);
	if(idepthbuf[i]<=id){ 
	  (pixelbuf[i])+=1;
	  idepthbuf[i]=d;
	}
    }
  }
  t.stop();
  t.print("idepth: ");
}
