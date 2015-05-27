#include <stdio.h>
#include <stdlib.h>
#include "MPIutils.hh"
#include "Timer.hh"
#include "vatoi.hh"
#include <unistd.h>
#include "Parameters.hh"

/* Description of what the hell this is doing:

   This tests to see if it is beneficial to overlap independent 
   sends in a pipelined fashion.  So for instance, you can 
   initiate a send and wait until later to see if the send 
   is completed. When you overlap a send, you initiate 
   another send while you are waiting for the first one 
   to complete.   
   
   nbuffer = Buffer depth.  The sends are overlapped.  
   This determines the amount of overlap/pipelining 
   between sends.  Default=2


 */

int main(int argc,char *argv[]){ 
  MPIenv *env = new MPIenv(argc,argv); // start MPI threads
  MPIcomm *comm = env->getComm(); // get communicator
  Timer timer;
  // **Now defined in Parameters.hh----------------------------
  //   int nbuffer = 2;
  //   int bufsize= 8*1024;; // 8k sizes (try different block sizes)
  //  int totaltransfer = 1024*1024 * 50; // 10 meg (1 gig) transfer
  timer.reset();
  if(argc>=2) bufsize = vatoi(argv[1]);
  if(argc>=3) nbuffer = atoi(argv[2]);
  int maxmsg = 4; // 4 maximum msgs per processor?
  float *sendbuf = new float[bufsize];
  typedef float *floatP;
  floatP *recvbuf= new floatP[nbuffer];
  typedef MPI_Request *MPI_RequestP;
  int max=totaltransfer/bufsize;
   MPI_RequestP rreq[2];
   MPI_Request *sreq = new MPI_Request[max];
   // need to know number of double-buffers
   if(!comm->rank()){
     for(int i=0;i<nbuffer;i++){
       rreq[i]    = new MPI_Request[comm->numProcs()];
       recvbuf[i] = new float[bufsize*comm->numProcs()];
     }
   }
   comm->barrier();
   timer.start(); // we are synchronized
   if(!comm->rank()){
     for(int i=0;i<comm->numProcs();i++){
       comm->iRecv(i,0,bufsize,(recvbuf[0])+bufsize*i,(rreq[0])[i]);
     }
   }
   comm->iSend(0,0,bufsize,sendbuf,sreq[0]);
   //-------------------------------------
   
   //fprintf(stderr,"totaltransfer=%u bufsize=%u max=%u\n",
   //   totaltransfer,bufsize,max);
   int n;
   for(n=1;n<max;n++){
     // post recieves first
     if(!comm->rank()){
       for(int i=0;i<comm->numProcs();i++){
	 comm->iRecv(i,n,bufsize,(recvbuf[n%nbuffer])+bufsize*i,(rreq[n%nbuffer])[i]); 
       }
     }
     if(!(n%maxmsg)) {
       MPI_Status stat;
       // collect all of our messages
       for(int i=0;i<maxmsg;i++)
	 comm->wait(sreq[i],stat);
     }
     comm->iSend(0,n,bufsize,sendbuf,sreq[n%maxmsg]);
     n--; // now collect from our last receive
     //*****************************************
     if(!comm->rank()){MPI_Status stat;
       for(int i=0;i<comm->numProcs();i++){
	 // number of processors
	 comm->wait((rreq[n%nbuffer])[i],stat);
       }
     }
     n++;
   }
   //---------------------------------------
   if(!comm->rank()){ MPI_Status stat;
    for(int i=0;i<comm->numProcs();i++){
     // number of processors
     comm->wait((rreq[(max-1)%nbuffer])[i],stat);
    }
   }
   {
     // collect remaining messages
     MPI_Status stat;
     for(int i=0;i<(n%maxmsg);i++) comm->wait(sreq[i],stat);
   }
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
   if(!comm->rank())
     timer.print();
   return 0;
}
