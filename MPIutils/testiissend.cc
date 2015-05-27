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
  // int nbuffer = 2;
  // int bufsize= 8*1024; // 8k sizes (try different block sizes)
  // int totaltransfer = 1024*1024 * 50; // 10 meg (1 gig) transfer
  timer.reset();
  if(argc>=2) bufsize = vatoi(argv[1]);
  if(argc>=3) nbuffer = atoi(argv[2]);
  
  float *sendbuf = new float[bufsize+0x00ffff];
  // align to page size
  float *asendbuf = (float*)((long)sendbuf & (~0x000000000000ffffl));
  asendbuf += 0x00ffff;
  sendbuf = asendbuf; // no deallocations, so this is OK 
  typedef float *floatP;
  floatP *recvbuf= new floatP[nbuffer];
  typedef MPI_Request *MPI_RequestP;
  //MPI_RequestP *rreq;
  //MPI_Request *sreq = new MPI_Request[nbuffer];
  MPI_RequestP rreq[2];
  MPI_Request sreq[2];
  // need to know number of double-buffers
  for(int j=0;j<bufsize;j++) sendbuf[j]=(float)j; // touch all 
  if(!comm->rank()){ 
    //  rreq = new MPI_RequestP[nbuffer];
    for(int i=0;i<nbuffer;i++){
      //fprintf(stderr,"assignbuf %u\n",i);
      rreq[i]    = new MPI_Request[comm->numProcs()];
      recvbuf[i] = new float[bufsize*comm->numProcs() + 0x00ffff];
      float *arecvbuf = (float*)((long)recvbuf[i] & (~0x000000000000ffffl));
      arecvbuf+=0x00ffff;
      recvbuf[i] = arecvbuf; // hell, no deallocations here
      // touch every element
      for(int j=0;j<bufsize*comm->numProcs();j++) arecvbuf[j]=(float)j;
    }
  }
  // all participate in the barrier
  comm->barrier();
  timer.start(); // we are synchronized
  // post recieves first
  if(!comm->rank()){
    for(int i=0;i<comm->numProcs();i++){
      // indexing backwards?  (rreq[i])[0] ??
      comm->iRecv(i,0,bufsize,(recvbuf[0])+bufsize*i,(rreq[0])[i]);
    }
  }
  comm->isSend(0,0,bufsize,sendbuf,sreq[0]);
  //-------------------------------------
  int max=totaltransfer/bufsize;
  if(!comm->rank()) 
	fprintf(stderr,"totaltransfer=%u bufsize=%u max=%u\n",
			    totaltransfer,bufsize,max);
  for(int n=1;n<max;n++){
    // post recieves first
    if(!comm->rank()){
      for(int i=0;i<comm->numProcs();i++){
	//fprintf(stderr,"[%u] n=%u recv bufnum=%u bufoffset=%u bufsize=%u i=%u\n",
	// comm->rank(),n,n%nbuffer,bufsize*i,bufsize,i);
	comm->iRecv(i,n%nbuffer,bufsize,(recvbuf[n%nbuffer])+bufsize*i,(rreq[n%nbuffer])[i]); 
	//fprintf(stderr,"[%u] good\n",comm->rank());
      }
    }
    //comm->barrier();
    //fprintf(stderr,"[%u] next barrier\r",comm->rank());
    // send data to processor 0
    comm->isSend(0,n%nbuffer,bufsize,sendbuf,sreq[n%nbuffer]);
    n--; // now collect from our last receive
    // waitfor requests to complete (how to reclaim them)
    /* need to compare this to
       -collecting/waiting on all handles simultaneously
       -waiting on each sendrequest on each cycle or waiting
       just once at the end
       */
    //*****************************************
    if(!comm->rank()){
      for(int i=0;i<comm->numProcs();i++){
	// number of processors
	MPI_Status stat;
	comm->wait((rreq[n%nbuffer])[i],stat);
      }
    }
    {
      MPI_Status stat;
      comm->wait(sreq[n%nbuffer],stat);
    }
    n++;
  }
  //---------------------------------------
  //fprintf(stderr,"\nfinal\n");
  // do our final collect
  //#if 0
  if(!comm->rank()){
    for(int i=0;i<comm->numProcs();i++){
      // number of processors
      MPI_Status stat;
      comm->wait((rreq[(max-1)%nbuffer])[i],stat);
    }
  }
  {
    MPI_Status stat;
    comm->wait(sreq[(max-1)%nbuffer],stat);
  }
  //#endif
  comm->barrier();
  timer.stop();
  if(!comm->rank()||comm->rank()==1) {
    printf("Thread[%u]: ",comm->rank());
    timer.print();
  }
  float rt,ut,st;
  timer.elapsedTimeSeconds(st,ut,rt);
  if(!comm->rank()) 
    printf("%f Mbytes/sec\n",
	   (float)(max*bufsize*sizeof(float)*comm->numProcs()) / 
	   (1024.0*1024.0*rt));
  delete env; // calls MPI_Finalize()
  return 0;
}
