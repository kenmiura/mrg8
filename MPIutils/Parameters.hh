#ifndef __PARAMS_H_
#define __PARAMS_H_

extern int nbuffer;  // For multibuffered codes, this is the buffering depth
// it represents the number of overlapping independent transfers per cycle.
extern int bufsize;  // Blocksize for each transfer
extern int totaltransfer; // total quantity of data transferred

#endif

