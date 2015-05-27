#ifndef __TIMER_HH_
#define __TIMER_HH_
#include <stdio.h>
#include "Timeval.hh"

class Timer {
  int running;
  double treal,tuser,tsystem;
#ifndef WIN32
  tms tm;
#endif
  timeval tv;
public:
  Timer() { reset(); }
  inline void reset(){
    treal=tuser=tsystem=0;
    running=0;
  }
  int start();
  int stop();
  void elapsedTimeSeconds(double &system,double &user,double &real);
  void elapsedTimeSeconds(float &system,float &user,float &real){
    double s,u,r;
    elapsedTimeSeconds(s,u,r);
    system=s; user=u; real=r;
  }
  double realTime();
  void print(char *str="",FILE *f=stderr){
    double s,u,r;
    this->elapsedTimeSeconds(s,u,r);
    fprintf(f,"Timer[%s]: real=%g user=%g system=%g\n",str,r,u,s);
  }
};

#endif
