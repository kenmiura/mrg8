#include "Timeval.hh"
#include <stdio.h>

void TimevalToDouble(timeval &tm,double &out){
  out=(double)tm.tv_sec + (double)tm.tv_usec/(double)1000000.0;
}

void DoubleToTimeval(double v,timeval &r){
  r.tv_sec = (int64_t)floor(v);
  r.tv_usec = (int64_t)((v - floor(v))*(double)1000000.0);
}

void SubtractTimeval(timeval &in1,timeval &in2,timeval &out){
  out.tv_sec = in1.tv_sec - in2.tv_sec;
  if(in1.tv_usec < in2.tv_usec){
    /* borrow */
    //  puts("\t\t   BORROW\n");
    out.tv_sec--;
    out.tv_usec = (int64_t)1000000l + in1.tv_usec - in2.tv_usec;
  }
  else out.tv_usec=in1.tv_usec-in2.tv_usec;
}

void AddTimeval(timeval &in1,timeval &in2,timeval &out){
  out.tv_sec = in1.tv_sec + in2.tv_sec;
  out.tv_usec = in1.tv_usec + in2.tv_usec;
  if(out.tv_usec>1000000l) {
    out.tv_sec+=1;
    out.tv_usec -= 1000000l;
  }
}

void CopyTimeval(timeval &in,timeval &out){
  out.tv_sec=in.tv_sec;
  out.tv_usec=in.tv_usec;
}

int SimpleCompareTimeval(timeval &in1,timeval &in2){
  /* if in1==in2, then return 0 */
  /* if in1>in2 then return 1 */
  /* if in1<in2 then return -1 */
  int r=0;
  if(in1.tv_sec < in2.tv_sec) {
    r=-1;
  }
  else { /* now we check for usec */
    if(in1.tv_sec == in2.tv_sec){
      if(in1.tv_sec < in2.tv_sec) {
	r=-1;
      }
      else {
	if(in1.tv_sec == in2.tv_sec){
	  return 0;
	}
	else
	  r=1;
      }
    }
    else {
      r=1;
    }
  }
  return r;
}

int CompareTimeval(timeval &in1, timeval &in2){
  /* if in1==in2, then return 0 */
  /* if in1>in2 then return 1 */
  /* if in1<in2 then return -1 */
  /* if in1 is > 2x in2, then return +2 */
  /* if in1 is < 1/2 in2, then return -2 */
  /* now we check for factor of 2 diff */
  timeval difference;
  int r = SimpleCompareTimeval(in1,in2);
  if(r==0) return 0;
  if(r>0){
    /* normalize the interval */
    SubtractTimeval(in1,in2,difference);
    if(SimpleCompareTimeval(difference,in2)>0) return -2;
    else return -1;
  }
  else { /* r<0 */
    SubtractTimeval(in2,in1,difference);
    if(SimpleCompareTimeval(difference,in1)>0) return 2;
    else return 1;
  }
}

void ZeroTimeval(timeval &t){
  t.tv_sec=0;
  t.tv_usec=0;
}

void PrintTimeval(char *lead_in,timeval &t){
  double d;
  TimevalToDouble(t,d);
  printf("%s: %u:%u --> %f\n",lead_in,t.tv_sec, t.tv_usec, d);
}

void GetTime(timeval &tm){
  gettimeofday(&tm,0);
}
