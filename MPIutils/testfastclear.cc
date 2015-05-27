#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "Timer.hh"

void main(int argc,char *argv[]){
  register int i;
  int blocksize=128;
  int bufsize=32*1024*1024; /* 32Meg buffer */
  char *buf= new char[bufsize];
  char *clrbuf = new char[bufsize];
	memset(clrbuf,0,bufsize);
  Timer t;

#if 0
	t.reset();
	t.start();
  for(i=0;i<100;i++){
	bzero(buf,bufsize);
  }
	t.stop();
	t.print("bzero: ");
#endif

	t.reset();
	t.start();
  for(i=0;i<100;i++){
	memset(buf,1,bufsize);
  }
	t.stop();
	t.print("memset: ");

	t.reset();
	t.start();
  for(i=0;i<100;i++){
	register int j;
	register char val=(char)i;
	register char *b=buf;
	for(j=0;j<bufsize;j++,b++) *b=val;
  }
	t.stop();
	t.print("loopset: ");

  blocksize=2048;
  for(int n=0;blocksize>32;n++,blocksize>>=1){
	printf("Blocksize=%u\n",blocksize);	
#if 0
	t.reset();
	t.start();
	for(i=0;i<100;i++){
		for(int block=0;block<bufsize; block+=blocksize){
			int blksz = (blocksize>(bufsize-block))?blocksize:(bufsize-block);
			bcopy(clrbuf,buf+block,blksz);
		}
	}
	t.stop();
	t.print("Blockbcpy: ");
#endif
	t.reset();
	t.start();
	for(i=0;i<100;i++){
		for(int block=0;block<bufsize; block+=blocksize){
			int remaining = bufsize-block;
			int blksz = (blocksize<remaining)?blocksize:remaining;
			memcpy(buf+block,clrbuf,blksz);
		}
	}
	t.stop();
	t.print("BlockMemcpy: ");
#if 0
	t.reset();
	t.start();
	for(i=0;i<100;i++){
		for(int block=0;block<bufsize; block+=blocksize){
			int blksz = (blocksize<(bufsize-block))?blocksize:(bufsize-block);
			register char *dst=buf+block,*src=clrbuf;
			for(int j=0;j<blksz;j++) *dst++=*src++;
		}
	}
	t.stop();
	t.print("Blockloopcpy:");
#endif
  }
}
