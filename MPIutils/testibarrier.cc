#include <stdio.h>
#include <stdlib.h>
#include "MPIutils.hh"
#include "Timer.hh"
#include "vatoi.hh"
#include <unistd.h>
#include "Parameters.hh"

int main(int argc,char *argv[]){ 
  MPIenv *env = new MPIenv(argc,argv); // start MPI threads
  MPIcomm *comm = env->getComm();
  Timer timer;
  // ** Now defined in Parameters.hh-------------------------------
  // int nbuffer = 2;
  // int bufsize= 8*1024;; // 8k sizes (try different block sizes)
  // int totaltransfer = 1024*1024 * 50; // 10 meg (1 gig) transfer
  MPI_Request *rreq,sreq;
  rreq = new MPI_Request[comm->numProcs()];
  timer.reset();
  if(argc>=2) bufsize = vatoi(argv[1]);
  if(argc>=3) nbuffer = atoi(argv[2]);
  if(comm->numProcs()<2){
    delete env;
    exit(0);
  }
  float *sendbuf = new float[bufsize];
  float *recvbuf = new float[bufsize*comm->numProcs()];
  // all participate in the barrier
  timer.start(); // we are synchronized
 // fprintf(stderr,"[%u] start\n",comm->rank());
  //-------------------------------------
  int max=totaltransfer/bufsize;
  if(!comm->rank()) fprintf(stderr,"totaltransfer=%u bufsize=%u max=%u\n",
	  totaltransfer,bufsize,max);
  for(int n=0;n<max;n++){
    if(!comm->rank()){
      for(int i=0;i<comm->numProcs()-1;i++){
	comm->iRecv(i+1,0,bufsize,recvbuf+bufsize*i,rreq[i]); 
	//comm->recv(i+1,bufsize,recvbuf+bufsize*i);
      }
    }
    else {
      //comm->send(0,bufsize,sendbuf);ls

      comm->iSend(0,0,bufsize,sendbuf,sreq);
    }
    MPI_Status stat;
    if(!comm->rank()) 
      for(int i=0;i<comm->numProcs()-1;i++)
	comm->wait(rreq[i],stat);
    else comm->wait(sreq,stat);
  }
  //---------------------------------------
  comm->barrier();
  timer.stop();
  if(!comm->rank() || comm->rank()==1) {
     printf("Thread[%u]: ",comm->rank());
     timer.print();
  }
  float rt,ut,st;
  timer.elapsedTimeSeconds(st,ut,rt);
  if(!comm->rank()) 
    printf("%f Mbytes/sec\n",
       (float)(max*bufsize*sizeof(float)*(comm->numProcs()-1)) /
	   (1024.0*1024.0*rt));
  delete env;
  return 0;
}
