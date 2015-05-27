#ifndef __TIMEVAL_HH_
#define __TIMEVAL_HH_

#include <sys/types.h>
#ifndef WIN32
#include <sys/times.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#else
#include <windows.h>
#define timeval tms;
#endif
#include <math.h>
#ifdef OSF
/* definitions for int64_t under OSF are in a wacky place */
#include <inttypes.h>
#endif

void TimevalToDouble(timeval &tm,double &out);
void DoubleToTimeval(double v,timeval &r);
void SubtractTimeval(timeval &in1,timeval &in2,timeval &out);
void AddTimeval(timeval &in1,timeval &in2,timeval &out);
void CopyTimeval(timeval &in,timeval &out);
int SimpleCompareTimeval(timeval &in1,timeval &in2);
int CompareTimeval(timeval &in1, timeval &in2);
void ZeroTimeval(timeval &t);
void PrintTimeval(char *lead_in,timeval &t);
void GetTime(timeval &tm);

/* Inlined stuff */
#define iSubtractTimeval(_in1,_in2,_out) {\
  _out.tv_sec = _in1.tv_sec - _in2.tv_sec;\
  if(_in1.tv_usec < _in2.tv_usec){\
    _out.tv_sec--;\
    _out.tv_usec = (int64_t)1000000l + _in1.tv_usec - _in2.tv_usec;\
  }\
  else _out.tv_usec=_in1.tv_usec-_in2.tv_usec;}

#define iTimevalToDouble(_tm,_out) _out=((double)_tm.tv_sec + (double)_tm.tv_usec/(double)1000000.0)

#define iGetTime(_tm) gettimeofday(&_tm,0);
#define iCopyTimeval(in,out) { (out).tv_usec=(in).tv_usec; (out).tv_sec=(in).tv_sec;}

#endif
