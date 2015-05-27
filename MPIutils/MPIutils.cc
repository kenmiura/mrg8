#include "MPIutils.hh"


void MPIbuffer::attach(int size){
  bufsize=size;
  char *mybuf = new char[bufsize];
  MPI_Buffer_attach(mybuf,bufsize);
}
void MPIbuffer::detach(){
  char *tmp;
  int size;
  MPI_Buffer_detach(&tmp,&size);
  if(tmp==mybuf) delete tmp;
  else MPI_Buffer_attach(tmp,size); // wasn't my buffer
}

MPIbuffer::MPIbuffer(int size):mybuf(0),bufsize(size){
  attach(size);
}
MPIbuffer::~MPIbuffer() {detach();}
void MPIbuffer::resize(int size){
  // check for pending requests using the buffer detach 
  // actually done by MPI_Buffer_detach() itself
  detach();
  attach(size);
}
void MPIbuffer::grow(int size){
  if(bufsize<size)
    resize(size);
}
void MPIbuffer::check(){ // make sure the buffersize is what we thought it was..
  char *tmp;
  int size;
  MPI_Buffer_detach(&tmp,&size);
  if(!tmp || tmp!=mybuf) 
    attach(bufsize);
  else if(tmp!=mybuf){
    if(size>=bufsize)
      MPI_Buffer_attach(tmp,size); // reattach
    else {
      delete tmp;
      attach(bufsize);
    }
  }
  else
    MPI_Buffer_attach(tmp,size); // reattach
  /* fails if existing buffer was too small and
     the source process deallocates the buffer without getting
     a handle to it.
     Can also fail if malloc != new on that architecture.
  */
}


// Must watch buffer attachment from external processes
MPIbuffer *MPIcomm::buffer;
MPI_Status MPIcomm::defstat;
