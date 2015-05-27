#ifndef WIN32
#include <unistd.h>
#else
#include <windows.h>
#endif

#if defined (AIX) || defined(LINUX)
#include <time.h>
#endif
#ifdef LINUX
#include <sys/time.h>
#endif

#include "Timer.hh"
#include <limits.h>
int Timer::start(){
  // struct timezone tz;
  if(running) return 0; // already running
  running=1;
#ifndef WIN32
  gettimeofday(&tv,0);
  times(&tm); // set current time
#else
  /* Something to get real time */
  /* Get Process Times Here using GetProcessTimes() */
#endif
  return 1;
}

int Timer::stop(){
  // timezone tz;
#ifndef WIN32
  tms tmc; // current time
  timeval tvc; // current realtime
  long _CLK_TCK = sysconf(_SC_CLK_TCK);
#endif
  if(!running) return 0; // already stopped
  running=0;
  // copy back current time
#ifndef WIN32
  gettimeofday(&tvc,0);
  times(&tmc);
  treal += ((double)(tvc.tv_sec - tv.tv_sec) + ((double)(tvc.tv_usec - tv.tv_usec))/((double)1000000.0));
 // tuser += (double)(tmc.tms_utime-tm.tms_utime)/_CLK_TCK;
  //tsystem += (double)(tmc.tms_stime-tm.tms_stime)/_CLK_TCK;
#else
  tuser = treal = tsystem = 0;
#endif
  return 1;
}

void Timer::elapsedTimeSeconds(double &system,double &user,double &real){
  int wasrunning=running;
  stop();
  system=tsystem;
  user=tuser;
  real=treal;
  if(wasrunning) start();
}

double Timer::realTime(){
  // timezone tz;
  double rt;
#ifndef WIN32
  tms tmc; // current time
  timeval tvc; // current realtime
#endif
  if(!running) return treal;
  gettimeofday(&tvc,0);
  //printf("treal=%f\n",treal);
  rt = ((double)(tvc.tv_sec - tv.tv_sec) + 
	  ((double)(tvc.tv_usec - tv.tv_usec))/((double)1000000.0));
  //printf("\tafter increment treal=%f\n",rt);
  return rt+treal;
}
