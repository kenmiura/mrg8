#include "MPIutils.hh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/ioctl.h>      /* necessary for baseline IOCTL */
//#define __USE_GNU 
#include <fcntl.h>          /* IO operations */

#ifdef HAS_LUSTRE
#include <lustre/lustre_user.h>
#endif
 //#undef __USE_GNU

/* need to redefine lseek to lseek64 if using 64-bit offsets */

#include "Timer.hh"
/* rules for building with Lustre IOCTLs */

#define Kilobytes(x) (x*1024l)
#define Megabytes(x) (x*1024l*1024l)
#define Gigabytes(x) (x*1024l*1024l*1024l)
#define MAX_LOCAL_SIZE Megabytes(32)

#define MPIIO

/***********************
class: PatternMPIIO
purpose: This is a base class I/O benchmark for all interleaved
        I/O benchmarks. The benchmarks all implement an interleaved
        parallel I/O pattern where each processor writes
        transfer sizes ranging from 64k to 7M in 64k increments
        in an interleaved pattern.  This kind of interleaved pattern
        is typical for parallel writes of 1D domain decomposed datasets
        such as particle lists or chunked 3D data.

        The "Transfersize" defines the write size for
        the strided pattern where each processor writes to 
            offset=rank*transfersize
        and writes an amount of
            size=transfersize
        and then will scan ahead by (nprocs-1)*tranfersize for the next write
        The test will write until file reaches "filesize" bytes long


        It virtualizes the following common
        steps in the benchmarking process.

    1) CreateFile(): opens up a file for writing.  The only important parameter is the name of the file.  There are options to pass Lustre striping information as well.

    2) WriteInteleaved(): This writes the interleaved pattern in parallel to disk.
         parameters are "transfersize" which defines the granularity of the interleaving
         and "filesize" which defines the total size of the file.  You must also give it
         a data array that is greater or equal to the size of the transfersize.

    3) CloseFile(): this implements the benchmark-specific file close operation.
    4) DeleteFile(): Deletes the file from the disk if it exists.

That's pretty much the only steps these classes virtualize.  There are internal helper functions,
but the above three steps are the key for very low-tech parallel I/O benchmarking.

If you look at the top of main(), you'll see a number of class constructors that write into
the "pio" object, which is of the base class type. You select different Pattern I/O benchmark
implementations by simply uncommenting the desired class when you allocate the pio object.
So for example, if you want to perform an MPI-IO collective test, simply uncomment the line
that reads

      pio = new PatternMPIIOcollective(comm);

and all of the testing will use the MPI-IO library with collective calls.  The benchmark
will still write the same strided pattern to disk, but just use a different API to do so.

extended by: PatternPOSIX, PatternMPIIO
************************/
// base class for all of the IO tests
struct PatternIO { 
protected:
  MPIcomm *comm;
  // *almost* pure virtual
public:
  // ********* Constructor
  PatternIO(MPIcomm *c):comm(c){}
  // ******** Create a new file (and set striping if necessary)
  virtual void CreateFile(char *filename,int nstripes=0,int stripesize=-1)=0;
  // ******* Close file and reclaim any associated resources
  virtual void CloseFile()=0; // use internal file descriptor
  // ******* Delete file (remove it from disk)
  void DeleteFile(char *filename) {
    if(unlink(filename)!=0){
      fprintf(stderr,"could not delete file [%s]\n",filename);
    }
  }
  // ******** Write an interleaved pattern into the file 
  // *  'Transfersize' defines the processor's local data write size for
  // *  the strided pattern where each processor writes to 
  // *      offset=rank*transfersize
  // *  and writes an amount of
  // *      size=transfersize
  // *  and then will scan ahead by (nprocs-1)*tranfersize for the next write
  // *  The test will write until file reaches "filesize" bytes long
  virtual void WriteInterleaved(void *data,long long transfersize,long long filesize)=0;
};


/***********************
class: PatternMPIIO
purpose: This is a base class I/O benchmark for interleaved 
         I/O using MPI-IO independent writes.  Independent
         writes disable the data-shipping and two-phase
         I/O performance features of the MPI-IO library.
      
behavior: this is a base class for the MPI-IO benchmarks that does not implement
      a working benchmark
uses: PatternIO
extended by: PatternMPIIOindependent, PatternMPIIOcollective
************************/
class PatternMPIIO : public PatternIO {
protected:
  MPI_File *file;
  MPI_Info info; // optional info (MPIIO hints)
public:
  PatternMPIIO(MPIcomm *c):PatternIO(c),file(0){}
  virtual ~PatternMPIIO(){if(file) this->CloseFile();}
  virtual void CreateFile(char *filename,int nstripes=0,int stripesize=-1){
    file=(MPI_File*)malloc(sizeof(MPI_File));
    comm->barrier();
    MPI_Info_create(&info);
#ifdef HAS_GPFS
    MPI_Info_set(info,"IBM_largeblock_io","true");
    MPI_File_set_info(*fh,info);
#endif
    MPI_File_open(comm->getCommunicator(),filename,MPI_MODE_CREATE|MPI_MODE_RDWR,
		  info,file);
    comm->barrier();
  }
  virtual void CloseFile(){
    if(file)
      MPI_File_close(file);
    free(file);
    file=0;
    MPI_Info_free(&info);
  }
  // WriteInterleaved is still pure virtual
  // so you cannot instantiate this class!!!
};


/***********************
class: PatternMPIIOindependent
purpose: Implements I/O benchmark for interleaved 
         I/O using MPI-IO independent writes.  Independent
         writes disable the data-shipping and two-phase
         I/O performance features of the MPI-IO library.
      
behavior: Performance is very bad, but it has not been
   tuned yet. 
      
uses: PatternMPIIO
************************/
class PatternMPIIOindependent : public PatternMPIIO {
protected:
public:
  PatternMPIIOindependent(MPIcomm *c):PatternMPIIO(c){}
  virtual void WriteInterleaved(void *data,long long transfersize,long long filesize){
    int globalsize=comm->nprocs()*transfersize,
      count=Gigabytes(2)/(comm->nprocs()*transfersize),
      blocklength=transfersize,stride=comm->nprocs()*transfersize,
      offset=comm->rank()*transfersize;
    long long fs=0;
    MPI_Offset foff=0;
    MPI_Status status;
    // first create a vector datatype
    MPI_Datatype elemtype=MPI_BYTE,chunktype;
    // MPI_Type_vector(count,blocklength,stride,elemtype,&chunktype);
    MPI_Type_create_subarray(1,&globalsize,&blocklength,&offset,MPI_ORDER_FORTRAN,elemtype,&chunktype);
    MPI_Type_commit(&chunktype);
    MPI_File_set_view(*file,0,elemtype,chunktype,"native",info);
    do {
      // To make it independent, we just do write_at instead of write_at_all
      MPI_File_write_at(*file,foff,data,transfersize,elemtype,&status);
      foff+=transfersize; fs+=(transfersize*comm->nprocs());
    } while(fs<filesize);
    MPI_Type_free(&chunktype);
  }
};

/***********************
class: PatternMPIIOcollective
purpose: Implements I/O benchmark for interleaved 
         I/O using MPI-IO collective writes. In theory
         the ROMIO layer should coalesce the many small
         interleaved writes into a more organized sequential
         write operations.
      
behavior: Performance is very bad, but it has not been
   tuned yet.  It is far slower than PatternPOSIX.
      
uses: PatternMPIIO
************************/
class PatternMPIIOcollective : public PatternMPIIO {
protected:
  // nothing new here
  // this is only collective by virtue of the MPI_File_write_at_all
public:
  PatternMPIIOcollective(MPIcomm *c):PatternMPIIO(c){}
  virtual void WriteInterleaved(void *data,long long transfersize,long long filesize){
    int globalsize=comm->nprocs()*transfersize,
      count=Gigabytes(2)/(comm->nprocs()*transfersize),
      blocklength=transfersize,stride=comm->nprocs()*transfersize,
      offset=comm->rank()*transfersize;
    long long fs=0;
    MPI_Offset foff=0;
    MPI_Status status;
    // first create a vector datatype
    MPI_Datatype elemtype=MPI_BYTE,chunktype;
    // MPI_Type_vector(count,blocklength,stride,elemtype,&chunktype);
    MPI_Type_create_subarray(1,&globalsize,&blocklength,&offset,MPI_ORDER_FORTRAN,elemtype,&chunktype);
    MPI_Type_commit(&chunktype);
    MPI_File_set_view(*file,0,elemtype,chunktype,"native",info);
    do {
      // this is the only thing that makes it collective (the write_at_all)
      MPI_File_write_at_all(*file,foff,data,transfersize,elemtype,&status);
      foff+=transfersize; fs+=(transfersize*comm->nprocs());
    } while(fs<filesize);
    MPI_Type_free(&chunktype);

  }
};

/***********************
class: PatternPOSIX
purpose: Implements I/O benchmark for interleaved 
         I/O using POSIX APIs to write to a shared file.
      
behavior: The POSIX benchmarks perform very well for transactions
      that are multiples of the OST stripe size. We were uncertain
      whether this was due to alignment issues or the size of the
      transactions. So two subclasses (PatternPOSIX_aligned, and
      PatternPOSIX_unique) were created to find out what is 
      primarily affecting performance.

      If you offset the start of the entire file by 64k that has
      a very negative effect on I/O performance. 
      Please see the WriteInterleaved() subroutine where
      there is an option to uncomment an alternative starting
      offset for the I/O transactions. Under this circumstance,
      the I/O performance is uniformly bad for all interleavings.

      However, the PatternPOSIXaligned subclass shows that
      even if all transactions are aligned to OST boundaries (implementing
      sparse parallel writes), the performance for transactions that
      match the OST stripe size are far faster, but the interleavings
      that are not an even multiple of the stripe width do just as
      poorly.  This indicates that Lustre is sensitive to alignment,
      but is *also* sensitive to the transaction size being precisely
      the same as the OST stripe size.
      
uses: PatternIO
subclassed by: PatternPOSIXaligned, PatternPOSIX_unique
************************/
class PatternPOSIX : public PatternIO {
protected:
  int fd;
  /* should time the lseek64 separately */
  virtual void WriteToOffset(void *data,long long offset,long long size){
    lseek(fd,offset,SEEK_SET);
    write(fd,data,size);
  }
  /* should time the lseek64 separately */
  virtual void WriteToOffsetChunked(void *data,long long offset,long long size){
    lseek(fd,offset,SEEK_SET);
    /* break transfer into 1mb pieces */
    while(size>Megabytes(1)){
      write(fd,data,Megabytes(1));
      size-=Megabytes(1);
    }
    if(size>0) write(fd,data,size);
  }
public:
  PatternPOSIX(MPIcomm *c):PatternIO(c),fd(0){} // technically fd0 is stdin, but we ignore that fact)
  virtual ~PatternPOSIX(){if(fd) this->CloseFile();}  
  virtual void CreateFile(char *filename,int nstripes=0,int stripesize=-1){
    /* compare O_WRONLY to O_RDWR */
    /* regular file open/create */
    int fd_oflag = O_CREAT | O_RDWR;
    
#ifndef HAS_LUSTRE
    fd = open(filename, fd_oflag, 0664);
    if(fd<0){
      fprintf(stderr,"Process[%u]: failed to open file [%s]\n",
	      comm->rank(),filename);
    }
#else
    if(comm->rank() == 0){
      /* create first and then barrier */
      struct lov_user_md opts = { 0 };
      /* Setup Lustre IOCTL striping pattern structure */
      opts.lmm_magic = LOV_USER_MAGIC;
      if(stripesize==-1) stripesize=Megs(1);
      opts.lmm_stripe_size = stripesize;
      if(nstripes==-1) nstripes=80; /* max */
      else if(nstripes==0) nstripes=4; /* default */
      opts.lmm_stripe_count = nstripes;
      
      /* File needs to be opened O_EXCL because we cannot set
	 Lustre striping information on a pre-existing file. */
      /* O_EXCL is error if O_CREAT and file already exists */
      /* could stat file to ensure it doesn't already exist */
      /* and then do an unlink */
      fd_oflag |= O_CREAT | O_EXCL | O_RDWR | O_LOV_DELAY_CREATE;
      fd = open(testFileName, fd_oflag, 0664);
      if(ioctl(fd, LL_IOC_LOV_SETSTRIPE, &opts)){
	fprintf(stderr, "\nError on ioctl for [%s]\n",
		filename);
      }
      comm->barrier();
    }
    else {
      comm->barrier();
      fd = open(filename, fd_oflag, 0664);
    }
#endif
    comm->barrier();
  }
  virtual void CloseFile(){
    if(fd)
      close(fd);
    fd=0;
  }
  // variable transfer size
  virtual void WriteInterleaved(void *data,long long transfersize,long long filesize){
    /* write transfersize and then skip by nprocs * transfersize */
    /* need to start at correct offset in file though:  seek past end? */
    // long long offset = (long long)comm->proc()*transfersize + kilobytes(64); /* unaligned test */
    long long offset = (long long)comm->proc()*transfersize;
    do {
      WriteToOffset(data,offset,transfersize); /* If transfersize > OST stripe size, 
						     lets pad things to nearest OST-aligned boundary */
      offset+=((long long)comm->nprocs() * transfersize); /* default unaligned case */
    } while(offset<filesize);
  }
};

/***********************
class: PatternPosixaligned
purpose: Implements I/O benchmark for interleaved 
         I/O.  It builds on the Standard PatternPOSIX 
         interleaving pattern, but ensures that each
         transaction is always aligned to Lustre
         OST stripe boundaries if the size of the 
         transaction exceeds the native stripe size.
         This simulates the HDF5 object alignment
         feature when writing in chunked mode.
         H5P_set_alignment();
behavior: 
      On Lustre, the PatternPOSIXaligned subclass shows that
      even if all transactions are aligned to OST boundaries (implementing
      sparse parallel writes), the performance for transactions that
      match the OST stripe size are far faster, but the interleavings
      that are not an even multiple of the stripe width do very
      poorly.  This indicates that Lustre is sensitive to alignment,
      but is *also* sensitive to the transaction size being precisely
      the same as the OST stripe size.
uses: PatternPOSIX

************************/
class PatternPOSIXaligned : public PatternPOSIX {
public:
  PatternPOSIXaligned(MPIcomm *c):PatternPOSIX(c){}
  // variable transfer size
  // next-try coordinate with neighbor approach
  // coord with neighbor with short writes
  // This virtual member overrides the version in the base class
  virtual void WriteInterleaved(void *data,long long transfersize,long long filesize){
    /* write transfersize and then skip by nprocs * transfersize */
    /* need to start at correct offset in file though:  seek past end? */
    long long offset;;
    long long aligned_transfersize;
    long long  nmul = transfersize/(Megabytes(1));
    if(nmul*Megabytes(1) < transfersize) nmul++; /* pad out to even multiple of transfer sizes */
    aligned_transfersize = (long long)nmul * Megabytes(1); /* multiple of the stripe size */
    if(!comm->rank())
      printf("%u ",(transfersize>Megabytes(1))?aligned_transfersize:transfersize);
    // now compute initial offset
    if(transfersize>Megabytes(1))/* if write is larger than OST size */
      offset = (long long)comm->proc()*aligned_transfersize; /* aligned case */
    else 
      offset = (long long)comm->proc()*transfersize; /* default unaligned case */
    do {
      // WriteToOffsetChunked(fd,data,offset,transfersize); /* If transfersize > OST stripe size, pad to nearest OST-aligned boundary */
      WriteToOffset(data,offset,transfersize); /* If transfersize > OST stripe size, pad to nearest OST-aligned boundary */
      if(transfersize > Megabytes(1))
	offset+=((long long)comm->nprocs() * aligned_transfersize); /* aligned case (sparse write) */
      else
	offset+=((long long)comm->nprocs() * transfersize); /* default unaligned case */
    } while(offset<filesize);
  }
};

/***********************
class: PatternPOSIX_unique
purpose: Implements I/O benchmark for degenerate 
         one-file-per-processor case.  This sets the
         "speed of light" for I/O performance.  We
         would like the writes to shared files to
         meet or exceed this performance. The only
         source of degradation should be the cost
         of sending small transactions to disk.
behavior: achieves performance that is close to the expected filesystem maximum
uses: PatternPOSIX
        
************************/

class PatternPOSIX_unique : public PatternPOSIX {
public:
  PatternPOSIX_unique(MPIcomm *c):PatternPOSIX(c){} // technically fd0 is stdin, but we ignore that fact)
  virtual ~PatternPOSIX_unique(){if(fd) this->CloseFile();}  
  virtual void CreateFile(char *filename,int nstripes=0,int stripesize=-1){
    /* compare O_WRONLY to O_RDWR */
    /* regular file open/create */
    char *rfilename = (char*)malloc(strlen(filename)+32);
    int fd_oflag = O_CREAT | O_RDWR;
    sprintf(rfilename,"%s%03u",filename,comm->rank());
    fd = open(rfilename, fd_oflag, 0664);
    if(fd<0){
      fprintf(stderr,"Process[%u]: failed to open file [%s]\n",
	      comm->rank(),rfilename);
    }
    free(rfilename);
    comm->barrier();
  }
  // variable transfer size
  // next-try coordinate with neighbor approach
  // coord with neighbor with short writes
  // This virtual member overrides the version in the base class
  virtual void WriteInterleaved(void *data,long long transfersize,long long filesize){
    /* writeInterleaved is degenerate in this case because we are writing one file per processor */
    /* all we are doing in this case is writing to independent files */
  /* write transfersize and then skip by nprocs * transfersize */
    /* need to start at correct offset in file though:  seek past end? */
    // long long offset = (long long)comm->proc()*transfersize + kilobytes(64); /* unaligned test */
    long long offset = (long long)comm->proc()*transfersize;
    long long base=0;
    do {
      WriteToOffset(data,base,transfersize); /* If transfersize > OST stripe size, 
						     lets pad things to nearest OST-aligned boundary */
      offset+=((long long)comm->nprocs() * transfersize); /* default unaligned case */
      base+=transfersize;
    } while(offset<filesize); // still use offset in the shared file case to determine cut-off for test
  }
};

int main(int argc, char *argv[]){
  const int ngb=20,ntrials=5;
  long long i,transfer_size;
  int stripesize;
  char filename[256];
  char *data;
  Timer t,tsync;
  MPIenv env(argc,argv);
  MPIcomm *comm=env.getComm();

  PatternIO *pio=0;
  // choose your testing scenario by uncommenting the correct IO implementation
  //pio = new PatternMPIIOindependent(comm);
  //pio = new PatternMPIIOcollective(comm);
  pio = new PatternPOSIX(comm);
  //pio = new PatternPOSIXaligned(comm);
  //pio = new PatternPOSIX_unique(comm);

  sprintf(filename,"interleaved_file"); // in case we use argv[1] in the future
  data = (char*)valloc(MAX_LOCAL_SIZE); // make sure data buffer is page aligned
  if(!data) {
	fprintf(stderr,"failed to malloc data on proc[%u]\n",comm->rank());
	exit(0);
  }
  { // push the stack
    char id = (char)(comm->rank() % 256);
    memset(data,id,MAX_LOCAL_SIZE); // init the data to relate to procID for testing
  } // pop stack
  comm->barrier();
  if(!comm->rank()) printf("Startup Proc[%u]\n",comm->rank()); // just announce for rank[0]

  // Start the patternIO benchmarking
  // The outer loop iterates through all interleavings from 64k to MAX buffer size
  // in increments of 64kb.
  for(transfer_size=Kilobytes(64);transfer_size<=MAX_LOCAL_SIZE;transfer_size+=Kilobytes(64)){
    double mintime,tskew,ttotal;
    int trial;
    // if(!comm->rank()) printf("Transfer size %u\n",(unsigned)transfer_size); // debug

    // Perform more than one trial to account for variability on shared system
    for(trial=0;trial<ntrials;trial++){
      long long filesize=Gigabytes(ngb);
      double filesizeGB=(double)ngb;
      //if(!comm->rank()) printf("\ttrial[%u]\n",trial); // debug
      // Create a new file (options for nstripes and stripesize are only if Lustre enabled)
      // Otherwise the 40 and Megabytes(1) are ignored
      pio->CreateFile(filename,40,Megabytes(1));
      comm->barrier();
      // Start the timers
      t.reset(); t.start(); // 't' is a local timer (measured before barrier) to see how much timing skew there is between processors 
      tsync.reset(); tsync.start(); // tsync measures time after all processors have finished
      // write to the interleaved file with currently selected transfersize (outer loop)
      pio->WriteInterleaved(data,transfer_size,filesize);
      // close file to ensure it is synced before we stop the timers
      pio->CloseFile(); // we make sure file is synced by closing before timing it
      t.stop(); /* stop timing before sync to find local performance */
      comm->barrier();
      tsync.stop(); /* stop timing after barrier to measure skew */
      if(trial<=1) { mintime=tsync.realTime();} // skip the first trial (block allocator wierdness)
      else if(mintime<tsync.realTime()) { mintime=tsync.realTime();}
      /* only delete if root processor */
      if(!comm->rank()) 
	pio->DeleteFile(filename); // only processor 0 deletes the file
      comm->barrier(); // make sure we don't get ahead of ourselves before creating next file
      if(!comm->rank())
	printf("%u %5.4f\n",(unsigned)transfer_size,(float)(filesizeGB/tsync.realTime()));
	// printf("\tsync=%5.4f\tt=%5.4f\n",(float)tsync.realTime(),(float)t.realTime());
    }
  }
  return 1;
}

