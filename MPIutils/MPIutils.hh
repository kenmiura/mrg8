#ifndef __MPIUTILS_HH_
#define __MPIUTILS_HH_
/* #include "Arch.hh" */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

typedef int Int32;
typedef long long Int64;
#define MPI_INT32 MPI_INT

class MPIbuffer {
  char *mybuf;
  int bufsize;
  void attach(int size);
  void detach();
public:
  MPIbuffer(int size);
  ~MPIbuffer();
  void resize(int size);
  void grow(int size);
  void check();
};

/*
class MPIthread{
public:
	MPI_Comm createCommunicator();
	int getRank();
	int growBufferTo(int size);
	int snapBufferTo(int size);
	int getTypeSize(MPI_Datatype type);
};
*/

/*
  Class: MPIcomm
  Purpose: Encapsulates rank and the communicator for
  a set of processes so that you don't have to remember
  as many parameters to each of your MPI calls.  It also
  takes advantage of C++ operator overloading to automatically
  choose the correct MPItype for buffers (further reducing
  the number of params you have to remember). 

  This is just a "convenience class" which otherwise
  does very little itself.  Consequently nearly 
  everything is defined in the header.  There it is
  automatically inlined so that it doesn't incurr 
  any overhead compared with using MPI C calls.
  In the case of invariant quantities like
  MPI_Comm_size and MPI_Comm_rank, you can actually
  eliminate the calling overhead by storing these
  values in the header and inlining the accessors
  to those private variables.
 */
/* Everything is inlined... 
           . . .this is a wrapper after all */

class MPIcomm {
  int mypid;
  int _nprocs;
  MPI_Comm comm;
  int default_tag;
  static MPIbuffer *buffer;
public:
  static MPI_Status defstat;
  MPIcomm(MPI_Comm commworld = MPI_COMM_WORLD):
    comm(commworld),default_tag(0){
      // since these don't change, optimize by assigning
      // to class variables to reduce subroutine call overhead
      MPI_Comm_rank(comm,&mypid);
      MPI_Comm_size(comm,&_nprocs);
  }
  inline MPI_Comm getCommunicator(){ return comm;}
  inline double time(){return MPI_Wtime();}
  inline int numProcs() { return _nprocs; }
  inline int rank() { return mypid; }
  inline int proc() {return mypid;}
  inline int nprocs(){return _nprocs;}
  inline void setDefaultTag(int d){ default_tag=d; }
  inline int send(int dest,int tag,MPI_Datatype type,int nelem,void *data){
    return MPI_Send(data,nelem,type,dest,tag,comm);
  }
  inline int send(int dest,MPI_Datatype type,int nelem,void *data){
    return send(dest,default_tag,type,nelem,data);
  }
  inline int send(int dest,int nelem,float *data){
    return send(dest,default_tag,MPI_FLOAT,nelem,data);
  }
  inline int send(int dest,int nelem,double *data){
    return send(dest,default_tag,MPI_DOUBLE,nelem,data);
  }
  inline int send(int dest,int nelem,Int32 *data){
    return send(dest,default_tag,MPI_INT32,nelem,data);
  }
  inline int send(int dest,int nelem,char *data){
    return send(dest,default_tag,MPI_CHAR,nelem,data);
  }
  inline int recv(int src,int tag,MPI_Datatype type,int nelem,void *data,MPI_Status &s=MPIcomm::defstat){
    return MPI_Recv(data,nelem,type,src,tag,comm,&s);
  } 
  inline int recv(MPI_Datatype type,int nelem,void *data,MPI_Status &s=MPIcomm::defstat){
    return recv(MPI_ANY_SOURCE,MPI_ANY_TAG,type,nelem,data,s);
  }
  
  inline int recv(int src,int nelem,float *data,MPI_Status &s=MPIcomm::defstat){
    return recv(src,MPI_ANY_TAG,MPI_FLOAT,nelem,data,s);
  }
  inline int recv(int src,int nelem,double *data,MPI_Status &s=MPIcomm::defstat){
    return recv(src,MPI_ANY_TAG,MPI_DOUBLE,nelem,data,s);
  }
  inline int recv(int src,int nelem,Int32 *data,MPI_Status &s=MPIcomm::defstat){
    return recv(src,MPI_ANY_TAG,MPI_INT32,nelem,data,s);
  }
  inline int recv(int src,int nelem,char *data,MPI_Status &s=MPIcomm::defstat){
    return recv(src,MPI_ANY_TAG,MPI_CHAR,nelem,data,s);
  }
  
  inline int recv(int nelem,float *data,MPI_Status &s=MPIcomm::defstat){
    return recv(MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_FLOAT,nelem,data,s);
  }
  inline int recv(int nelem,double *data,MPI_Status &s=MPIcomm::defstat){
    return recv(MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_DOUBLE,nelem,data,s);
  }
  inline int recv(int nelem,Int32 *data,MPI_Status &s=MPIcomm::defstat){
    return recv(MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_INT32,nelem,data,s);
  }
  inline int recv(int nelem,char *data,MPI_Status &s=MPIcomm::defstat){
    return recv(MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_CHAR,nelem,data,s);
  }
  // Need support for 
  // S, B, R sends
  // Buffered: Need to preallocate or auto-alloc back buffers.
  // Synchronous: Blocks until matching receive posted
  // Ready: Error unless matching receive has already been posted.
  void setBufferSize(int nbytes){
    if(buffer) buffer->resize(nbytes);
    else buffer = new MPIbuffer(nbytes);
  }
  void setBufferSize(MPI_Datatype type,int nelem){
    // compute size
    int elemsize = 8;
    if(buffer) buffer->resize(nelem*elemsize);
    else buffer = new MPIbuffer(nelem*elemsize);
  }
  void growBufferSize(int nbytes){
    if(buffer) buffer->grow(nbytes);
    else buffer = new MPIbuffer(nbytes);
  }
  void growBufferSize(MPI_Datatype type,int nelem){
    // compute size
    int elemsize = 8;
    if(buffer) buffer->grow(nelem*elemsize);
    else buffer = new MPIbuffer(nelem*elemsize);
  }
  inline int bSend(int dest,int tag,MPI_Datatype type,int nelem,void *data){
    // need to get MPI size for the datatype
    if(!buffer) growBufferSize(nelem*8);
    return MPI_Bsend(data,nelem,type,dest,tag,comm);
  }
  inline int bSend(int dest,MPI_Datatype type,int nelem,void *data){
    return bSend(dest,default_tag,type,nelem,data);
  }
  
  inline int bSend(int dest,int nelem,float *data){
    return bSend(dest,default_tag,MPI_FLOAT,nelem,data);
  }
  inline int bSend(int dest,int nelem,double *data){
    return bSend(dest,default_tag,MPI_DOUBLE,nelem,data);
  }
  inline int bSend(int dest,int nelem,Int32 *data){
    return bSend(dest,default_tag,MPI_INT32,nelem,data);
  }
  inline int bSend(int dest,int nelem,char *data){
    return bSend(dest,default_tag,MPI_CHAR,nelem,data);
  }
  inline int iSend(int dest,int tag,MPI_Datatype type,int nelem,void *data,MPI_Request &req){
    // need to get MPI size for the datatype
    return MPI_Isend(data,nelem,type,dest,tag,comm,&req);
  }
  inline int iSend(int dest,MPI_Datatype type,int nelem,void *data,MPI_Request &req){
    return iSend(dest,default_tag,type,nelem,data,req);
  }
  
  inline int iSend(int dest,int tag,int nelem,float *data,MPI_Request &req){
    return iSend(dest,tag,MPI_FLOAT,nelem,data,req);
  }
  inline int iSend(int dest,int tag,int nelem,double *data,MPI_Request &req){
    return iSend(dest,tag,MPI_DOUBLE,nelem,data,req);
  }
  inline int iSend(int dest,int tag,int nelem,Int32 *data,MPI_Request &req){
    return iSend(dest,tag,MPI_INT32,nelem,data,req);
  }
  inline int iSend(int dest,int tag,int nelem,char *data,MPI_Request &req){
    return iSend(dest,tag,MPI_CHAR,nelem,data,req);
  }
  
  inline int iSend(int dest,int nelem,float *data,MPI_Request &req){
    return iSend(dest,default_tag,MPI_FLOAT,nelem,data,req);
  }
  inline int iSend(int dest,int nelem,double *data,MPI_Request &req){
    return iSend(dest,default_tag,MPI_DOUBLE,nelem,data,req);
  }
  inline int iSend(int dest,int nelem,Int32 *data,MPI_Request &req){
    return iSend(dest,default_tag,MPI_INT32,nelem,data,req);
  }
  inline int iSend(int dest,int nelem,char *data,MPI_Request &req){
    return iSend(dest,default_tag,MPI_CHAR,nelem,data,req);
  }

  inline int isSend(int dest,int tag,MPI_Datatype type,int nelem,void *data,MPI_Request &req){
    // need to get MPI size for the datatype
    return MPI_Issend(data,nelem,type,dest,tag,comm,&req);
  }
  inline int isSend(int dest,MPI_Datatype type,int nelem,void *data,MPI_Request &req){
    return isSend(dest,default_tag,type,nelem,data,req);
  }
  
  inline int isSend(int dest,int tag,int nelem,float *data,MPI_Request &req){
    return isSend(dest,tag,MPI_FLOAT,nelem,data,req);
  }
  inline int isSend(int dest,int tag,int nelem,double *data,MPI_Request &req){
    return isSend(dest,tag,MPI_DOUBLE,nelem,data,req);
  }
  inline int isSend(int dest,int tag,int nelem,Int32 *data,MPI_Request &req){
    return isSend(dest,tag,MPI_INT32,nelem,data,req);
  }
  inline int isSend(int dest,int tag,int nelem,char *data,MPI_Request &req){
    return isSend(dest,tag,MPI_CHAR,nelem,data,req);
  }
  
  inline int isSend(int dest,int nelem,float *data,MPI_Request &req){
    return isSend(dest,default_tag,MPI_FLOAT,nelem,data,req);
  }
  inline int isSend(int dest,int nelem,double *data,MPI_Request &req){
    return isSend(dest,default_tag,MPI_DOUBLE,nelem,data,req);
  }
  inline int isSend(int dest,int nelem,Int32 *data,MPI_Request &req){
    return isSend(dest,default_tag,MPI_INT32,nelem,data,req);
  }
  inline int isSend(int dest,int nelem,char *data,MPI_Request &req){
    return isSend(dest,default_tag,MPI_CHAR,nelem,data,req);
  }
  
  inline int ibSend(int dest,int tag,MPI_Datatype type,int nelem,void *data,MPI_Request &req){
    // need to get MPI size for the datatype
    if(!buffer) growBufferSize(nelem*8);
    return MPI_Ibsend(data,nelem,type,dest,tag,comm,&req);
  }
  inline int ibSend(int dest,MPI_Datatype type,int nelem,void *data,MPI_Request &req){
    return ibSend(dest,default_tag,type,nelem,data,req);
  }
  inline int iRecv(int src,int tag,MPI_Datatype type,int nelem,void *data,MPI_Request &req){
    return MPI_Irecv(data,nelem,type,src,tag,comm,&req);
  }
  inline int iRecv(MPI_Datatype type,int nelem,void *data,MPI_Request &req){
    return iRecv(MPI_ANY_SOURCE,MPI_ANY_TAG,type,nelem,data,req);
  }
  
  inline int iRecv(int src, int tag, int nelem,float *data,MPI_Request &req){
    return iRecv(src,tag,MPI_FLOAT,nelem,data,req);
  }
  inline int iRecv(int src, int tag, int nelem,double *data,MPI_Request &req){
    return iRecv(src,tag,MPI_DOUBLE,nelem,data,req);
  }
  inline int iRecv(int src, int tag, int nelem,Int32 *data,MPI_Request &req){
    return iRecv(src,tag,MPI_INT32,nelem,data,req);
  }
  inline int iRecv(int src, int tag, int nelem,char *data,MPI_Request &req){
    return iRecv(src,tag,MPI_CHAR,nelem,data,req);
  }
  
  inline int iRecv(int nelem,float *data,MPI_Request &req){
    return iRecv(MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_FLOAT,nelem,data,req);
  }
  inline int iRecv(int nelem,double *data,MPI_Request &req){
    return iRecv(MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_DOUBLE,nelem,data,req);
  }
  inline int iRecv(int nelem,Int32 *data,MPI_Request &req){
    return iRecv(MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_INT32,nelem,data,req);
  }
  inline int iRecv(int nelem,char *data,MPI_Request &req){
    return iRecv(MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_CHAR,nelem,data,req);
  }
  // waiting routines
  int wait(MPI_Request &req,MPI_Status &stat){
    return MPI_Wait(&req,&stat);
  }
  int test(MPI_Request &req,MPI_Status &stat,int &flag){
    return MPI_Test(&req,&flag,&stat);
  }
  int requestFree(MPI_Request &req){
    return MPI_Request_free(&req);
  }
  int waitAny(int nrequests,MPI_Request *requestarray,int &completed,MPI_Status &stat){
    return MPI_Waitany(nrequests,requestarray,&completed,&stat);
  }
  int waitAll(int nreq, MPI_Request *reqarray,MPI_Status *statarray){
    return MPI_Waitall(nreq,reqarray,statarray);
  }
  int probe(int source,int tag,MPI_Status &stat){
    return MPI_Probe(source,tag,comm,&stat);
  }
  int probe(int &flag,MPI_Status &stat){
    return MPI_Probe(MPI_ANY_SOURCE,MPI_ANY_TAG,comm,&stat);
  }
  int iProbe(int source,int tag,int &flag,MPI_Status &stat){
    return MPI_Iprobe(source,tag,comm,&flag,&stat);
  }
  int iProbe(int &flag,MPI_Status &stat){
    return MPI_Iprobe(MPI_ANY_SOURCE,MPI_ANY_TAG,comm,&flag,&stat);
  }
  //-collective comm--
  int gather(int root,MPI_Datatype type,int localnelem,void *senddata,void *recvdata){
    return MPI_Gather(senddata,localnelem,type,recvdata,localnelem,type,root,comm);
  }
  
  int gather(int root,int localnelem,float *senddata,float *recvdata){
    return MPI_Gather(senddata,localnelem,MPI_FLOAT,recvdata,localnelem,MPI_FLOAT,root,comm);
  }
  int gather(int root,int localnelem,double *senddata,double *recvdata){
    return MPI_Gather(senddata,localnelem,MPI_DOUBLE,recvdata,localnelem,MPI_DOUBLE,root,comm);
  }
  int gather(int root,int localnelem,Int32 *senddata,Int32 *recvdata){
    return MPI_Gather(senddata,localnelem,MPI_INT32,recvdata,localnelem,MPI_INT32,root,comm);
  }
  int gather(int root,int localnelem,char *senddata,char *recvdata){
    return MPI_Gather(senddata,localnelem,MPI_CHAR,recvdata,localnelem,MPI_CHAR,root,comm);
  }
  
  int gatherv(int root,MPI_Datatype type,
	      int localnelem,int *globalnelem,int *displacements,
	      void *senddata,void *recvdata){
    return MPI_Gatherv(senddata,localnelem,type,
		       recvdata,globalnelem,displacements,type,root,comm);
  }
  int allgather(MPI_Datatype type,int nelem,void *senddata,void *recvdata){
    return MPI_Allgather(senddata,nelem,type,recvdata,nelem,type,comm);
  }
  int alltoall(MPI_Datatype type,int nelem,void *senddata,void *recvdata){
    return MPI_Alltoall(senddata,nelem,type,recvdata,nelem,type,comm);
  }
  int scatter(int root,MPI_Datatype type,int localnelem,void *senddata,void *recvdata){
    return MPI_Scatter(senddata,localnelem,type,recvdata,localnelem,type,root,comm);
  }
  int scatter(int root,int localnelem,float *senddata,float *recvdata){
    return MPI_Scatter(senddata,localnelem,MPI_FLOAT,recvdata,localnelem,MPI_FLOAT,root,comm);
  }
  int scatter(int root,int localnelem,double *senddata,double *recvdata){
    return MPI_Scatter(senddata,localnelem,MPI_DOUBLE,recvdata,localnelem,MPI_DOUBLE,root,comm);
  }
  int scatter(int root,int localnelem,Int32 *senddata,Int32 *recvdata){
    return MPI_Scatter(senddata,localnelem,MPI_INT32,recvdata,localnelem,MPI_INT32,root,comm);
  }
#ifdef T3E
  int scatter(int root,int localnelem,int *senddata,Int32 *recvdata){
    return MPI_Scatter(senddata,localnelem,MPI_INT,recvdata,localnelem,MPI_INT,root,comm);
  }
#endif
  int scatter(int root,int localnelem,char *senddata,char *recvdata){
    return MPI_Scatter(senddata,localnelem,MPI_CHAR,recvdata,localnelem,MPI_CHAR,root,comm);
  }
  int scatterv(int root,MPI_Datatype type,
	       int localnelem,int *globalnelem,int *displacements,
	       void *senddata,void *recvdata){
    return MPI_Scatterv(senddata,globalnelem,displacements,type,
		       recvdata,localnelem,type,root,comm);
  }
  int bcast(int root,MPI_Datatype type,int nelements,void *data){
    return MPI_Bcast(data,nelements,type,root,comm);
  }
  int bcast(int root,int nelements,float *data){
    return bcast(root,MPI_FLOAT,nelements,data);
  }
  int bcast(int root,int nelements,double *data){
    return bcast(root,MPI_DOUBLE,nelements,data);
  }
  int bcast(int root,int nelements,Int32 *data){
    return bcast(root,MPI_INT32,nelements,data);
  }  
#ifdef T3E
  int bcast(int root,int nelements,int *data){
    return bcast(root,MPI_INT,nelements,data);
  } 
  int bcast(int root,int &data){
    return bcast(root,1,&data);
  }
#endif
  int bcast(int root,Int32 &data){
    return bcast(root,1,&data);
  } 
  int barrier(){
    return MPI_Barrier(comm);
  }
  //double time(){
  //  return MPI_Wtime();
  //}
  // Lets play with UserReduce() collective operations.
  
  //------User Reduce Function------------------
  typedef void (*UserReduceFunction)(void *in,void *out,int *len,MPI_Datatype *dt);
  MPI_Op myOp;
  MPI_Op createCommutativeOp(UserReduceFunction reduceOp){
    MPI_Op_create(reduceOp,true,&myOp);
    return myOp;
  }
  MPI_Op createNoncommutativeOp(UserReduceFunction reduceOp){
    MPI_Op_create(reduceOp,false,&myOp);
    return myOp;
  }
  void releaseOp(MPI_Op &handle){
    MPI_Op_free(&handle);
  }
  void freeOp(MPI_Op &handle){ releaseOp(handle);}
  void releaseOp(){
    MPI_Op_free(&myOp);
  }
  int reduce(int root,void *datain,void *dataout,int len,MPI_Op op,MPI_Datatype datatype){
     return MPI_Reduce(datain,dataout,len,datatype,op,root,comm);
  }
  int reduce(int root,float *datain,float *dataout,int len,MPI_Op op){
    return MPI_Reduce(datain,dataout,len,MPI_FLOAT,op,root,comm);
  }
  int reduce(int root,Int32 *datain,Int32 *dataout,int len,MPI_Op op){
    return MPI_Reduce(datain,dataout,len,MPI_INT32,op,root,comm);
  }
  int reduce(int root,float *datain,float *dataout,int len){
    return MPI_Reduce(datain,dataout,len,MPI_FLOAT,myOp,root,comm);
  }
  int reduce(int root,Int32 *datain,Int32 *dataout,int len){
    return MPI_Reduce(datain,dataout,len,MPI_INT32,myOp,root,comm);
  }
  int reduce(int root,char *datain,char *dataout,int len){
    return MPI_Reduce(datain,dataout,len,MPI_CHAR,myOp,root,comm);
  }
  //----MPI Topology directives (just the cartesian stuff)
  MPIcomm *cartCreate(int ndims=3){
    MPI_Comm dcom;
    int pdims[5]={0,0,0,0,0}; // who would use more than 5D?
    int pperiods[5]={0,0,0,0,0};
    MPI_Dims_create(this->_nprocs,ndims,pdims);
    MPI_Cart_create(comm,ndims,pdims,pperiods,1/*reorder true*/,&dcom);
    return new MPIcomm(dcom); // create a new communication object
    // dims might be reordered
  }
  void getProcLayout(int *proclayout,int ndims=3){
    int pdims[5]={0,0,0,0,0};
    MPI_Dims_create(this->_nprocs,ndims,pdims);
    for(int i=0;i<ndims;i++) proclayout[i]=pdims[i];
  }
  void getCoords(int *mycoords,int ndims=3){
    MPI_Cart_coords(comm,this->mypid,ndims,mycoords);
  }
  /*
    void myProd(void *in,void *out,int *len,MPI_Datatype *d);
    MPI_Op myOp;
    MPI_Datatype datatype = MPI_INT32;
    MPI_Op_create(myProd,True,&myOp);
    MPI_Reduce(input,result,len=100,datatype,myOp,root,comm);
   */
};


/*
  Class: MPIenv
  Purpose: Hides MPI_Init/Finalize operation.
  Allocating this object implicitly forks the MPI
  processes.  Destroying it implicitly destroys the
  processes.  Typically it would be used as a literal
  at the top of your program.  
     int main(int argc,char *argv[]){
        MPIenv mpi(argc,argv);
     }
     
  The destructor is called automatically if the
  program exits for any reason, so you don't 
  get as many annoying "code exited without calling
  MPI_Finalize()" messages and can be lazy about
  your exit/exception-handling strategy.

  A typical code (that actually does something) would
  require you get a communicator from the MPI 
  environment.     
     int main(int argc,char *argv[]){
        MPIenv mpi(argc,argv);
	 MPIcomm *comm = mpi.getComm();
	 comm->barrier(); // do something pointless...
     }
 */
/* MPI grouping mechanisms must be here in the MPIenv */
class MPIenv {
  MPIcomm *defaultcomm;
public:
  MPIenv(int &argc,char **&argv){
    MPI_Init(&argc,&argv);
    defaultcomm = new MPIcomm(MPI_COMM_WORLD);
  }
  ~MPIenv(){ delete defaultcomm; MPI_Finalize();}
  MPIcomm *getComm() { return defaultcomm; }
  MPIcomm *getComm(MPIcomm custom_communicator){
    return new MPIcomm(custom_communicator);
  }
};

/* The MPI-IO class is a work-in progress.
   This is not ready for prime-time because it
   is exposes and API that is as obtuse as the
   one it is hiding.

   Also, MPI-IO is problematic because it uses
   integers for offsets (not off64_t or something
   larger).  So there will be serious integer
   overflow problems.

   When this class is ready, it should really
   be a parallel I/O base class that hides 
   multiple underlying implementations, including
   MPI-POSIX. (hedge for 64-bit pointer issues)

*/

class ParallelIO {
  MPIcomm *comm;

  int hastype;
  MPI_Info info;
  MPI_Datatype elemtype,chunktype;
  MPI_File file;
  MPI_Offset foffset;
  void SetView(MPI_Offset disp){
    if(hastype)
      MPI_File_set_view(file,disp,elemtype,chunktype,"native",info);
  }
public:
  ParallelIO(MPIcomm *c,char *filename,int mode):comm(c),hastype(0){
    /* modes are MPI_MODE_CREATE, MPI_MODE_RDWR */
    MPI_Info_create(&info);
    MPI_File_open(comm->getCommunicator(),filename,mode,info,&file);
  }
  ~ParallelIO(){
    if(hastype) MPI_Type_free(&chunktype);
    hastype=0;
    MPI_Info_free(&info);
    MPI_File_close(&file);
  }
  void SetInfo(char *key,char *value){
    // MPI_Info_set(info,"IBM_largeblock_io","true");
    MPI_Info_set(info,key,value);
    MPI_File_set_info(file,info);
  }
  void SetPatternBlock(int ndims,int *globalsizes,int *localsizes,
	     int *startoffsets,MPI_Datatype elemtype_){
    elemtype=elemtype_;
    if(hastype) MPI_Type_free(&chunktype);
    hastype=1;
    MPI_Type_create_subarray(ndims,globalsizes,localsizes,startoffsets,
			     MPI_ORDER_FORTRAN,elemtype,&chunktype);
    MPI_Type_commit(&chunktype);
  }
  void SetPatternBlock1D(int globalsizes,int localsizes,int startoffsets,MPI_Datatype elemtype_){
    elemtype=elemtype_;
    if(hastype) MPI_Type_free(&chunktype);
    hastype=1;
    MPI_Type_create_subarray(1,&globalsizes,&localsizes,&startoffsets,
			     MPI_ORDER_FORTRAN,elemtype,&chunktype);
    MPI_Type_commit(&chunktype);
  }
  void SetPatternStrided(int count,int blocklength, int stride, MPI_Datatype elemtype_){
    elemtype=elemtype_;
    if(hastype) MPI_Type_free(&chunktype);
    hastype=1;
    MPI_Type_vector(count,blocklength,stride,elemtype,&chunktype);
    MPI_Type_commit(&chunktype);
  }
  void WriteCollective(long long offset,void *data,int length){
    MPI_Status status;
    MPI_File_write_at_all(file,
			  (MPI_Offset)offset, // offset is in elements (by view)
			  data,
			  length,
			  elemtype,&status);
 
  }
  void WriteIndependent(long long offset,void *data,int length){
    MPI_Status status;
    MPI_File_write_at(file,
			  (MPI_Offset)offset,
			  data,
			  length,
			  elemtype,&status);
 
  }
};

#endif // __MPIUTILS_HH_
