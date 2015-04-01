c  MRG8 for release. For benchmarking purposes only. 
c  National Institute of Informatics      K.Miura

Copyright (C) 2006, Kenichi Miura,   All rights reserved. 
                         
c  Redistribution and use in source and binary forms, with or without
c  modification, are permitted provided that the following conditions
c  are met:

c    1. Redistributions of source code must retain the above 
c       copyright notice, this list of conditions and the following 
c       disclaimer.
c
c    2. Redistributions in binary form must reproduce the above
c    copyright notice, this list of conditions and the following
c    disclaimer in the documentation and/or other materials provided
c    with the distribution.
c
c********************************************************************
c   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER  "AS IS", AND
c   ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED 
c   TO,THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
c   PARTICULAR PURPOSE ARE DISCLAIMED.
c   IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
c   FOR ANY DIRECT,INDIRECT, INCIDENTAL,SPECIAL, EXEMPLARY, OR
c   CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
c   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
c   OR PROFITS;OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY 
c   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
c   TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
c   OF THE USE OF HIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY 
c   OF SUCH DAMAGE.
c********************************************************************
c     integer*4 routine and real*8 routine separated
c      10-6-06   KM
c
c   frt -KV9 source.f --> hardware 8-byte integer ops.
      program cortest
      parameter(NN=80000000)
      real*8  RAN1(NN),RAN2(NN)
      integer*8 N,NN,ISEED,MASK
      integer*8 IRAN1(NN),IRAN2(NN)
      real*8 RNORM,SUM,TEMP
      real*8 t0,t1,time
      data MASK/2147483647/
      RNORM=1.0d0/dble(MASK)
      N=NN/2
      open(6,file='unit6',status='new')
c     write(6,*)"Modulus=",MASK,"   Normalization=",RNORM
c
c     measurement of mrg8n routine
c
      ISEED=13579
      call mrg8dn(ISEED,RAN1,100)
      call clock(t0,2,2)
      call mrg8dn(ISEED,RAN1,N)
      call clock (t1,2,2)
      time=t1-t0
      call clock(t0,2,2)
      call mrg8dn(ISEED,RAN1,NN)
      call clock (t1,2,2)
      SUM=0.d0
      do i=1,NN
      SUM=SUM+RAN1(i)
      enddo
      SUM=SUM/dfloat(NN)
      TEMP=0.d0
      do i=1,NN
      TEMP=TEMP+(RAN1(i)-SUM)**2
      enddo
      TEMP=TEMP/dfloat(NN-1)
      TEMP=dsqrt(TEMP)
      write(6,*)"Arithmetic mean:  ",SUM," vs. ",.5d0
      write(6,*)"Standard deviation: ", TEMP," vs. ", 
     x           dsqrt(1.d0/12.d0)
      write(6,*)"MRG8n: time per RNG  in microseconds"
      write(6,*) (t1-t0-time)/(NN-N)
      write(6,*) t1-t0,time,NN,N
c
c     measurement of the modified mrg8n routine
c

      ISEED=13579
      call mrg8dnz2(ISEED,RAN2,100)
      call clock(t0,2,2)
      call mrg8dnz2(ISEED,RAN2,N)
      call clock (t1,2,2)
      time=t1-t0
      call clock(t0,2,2)
      call mrg8dnz2(ISEED,RAN2,NN)
      call clock (t1,2,2)
      write(6,*)"MRG8nz: time per RNG  in microseconds"
      write(6,*) (t1-t0-time)/(NN-N)
      write(6,*) t1-t0,time,NN,N

      write(6,*)"First 100 random numbers:  mrg8dn vs mrg8dnz2"
      do k=1,100
      write(6,*) "(",k,")  ",RAN1(k),"   ",RAN2(k)
      enddo
c
      close(6)
      stop
      end
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine mrg8dn(iseed,ran,n)
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
Copyright (C) 2006, Kenichi Miura,   All rights reserved. 
c      dfp version.     integer output deleted.   9-15-06  KM
c****  Sample coding of 8th order Multiple Recursive RNG.
c      L'Ecuyer's polynomial.
      integer iseed,n
      real*8 ran(1)
      real*8 rnorm/4.6566128752458d-10/
      integer*4 iflag/0/
      integer*8 s,s1,s2,mask,a(8)
      integer*8 x(8)
      integer*4 xx(8)
      data a/1089656042,1906537547,1764115693,1304127872,
     x       189748160,1984088114,626062218,1927846343/
      data x/8*0/
      data mask/2147483647/
c
      if(iflag.eq.0) then
       if(iseed.eq.0)iseed=97531
       call mcg64ni(iseed,xx,8)
       write(6,*) "iseed=",iseed
       write(6,*)"First seeds X(k)"
       do k=1,8
       x(k)=ishft(xx(k),-1)
       write(6,*)"(",k,")  ",x(k)
       enddo
       iflag=1
      endif
      do i=1,n
      s1=a(1)*x(1)+a(2)*x(2)+a(3)*x(3)+a(4)*x(4)
      s2=a(5)*x(5)+a(6)*x(6)+a(7)*x(7)+a(8)*x(8)
      s=iand(s1,mask)+ishft(s1,-31)+iand(s2,mask)+ishft(s2,-31)
      s=iand(s,mask)+ishft(s,-31)
      x(8)=x(7)
      x(7)=x(6)
      x(6)=x(5)
      x(5)=x(4)
      x(4)=x(3)
      x(3)=x(2)
      x(2)=x(1)
      x(1)=iand(s,mask)+ishft(s,-31)
      ran(i)=dble(x(1))*rnorm
      enddo
      return
      end
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine mrg8dnz2(iseed,ran,n)
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
Copyright (C) 2006, Kenichi Miura,   All rights reserved. 
c     10-06-06   KM
c****  Sample coding of 8th order Multiple Recursive RNG
c****  A longer state vector to reduce the frequency of
c****  copying most recent 8 values.
c****  10-06-06   KM
      integer iseed,n
      real*8 ran(1)
      real*8 rnorm/4.6566128752458d-10/
      integer*4 iflag/0/
      integer*4 kmax/1024/
      integer*4 xx(8)
      integer*8 s,s1,s2,mask,a(8)
      integer*8 x(1032)
      data a/1089656042,1906537547,1764115693,1304127872,
     x       189748160,1984088114,626062218,1927846343/
      data x/1032*0/
      data mask/2147483647/
      if(iflag.eq.0) then
      if(iseed.eq.0)iseed=97531
       write(6,*)"kmax=",kmax
       call mcg64ni(iseed,xx,8)
       write(6,*)"iseed=",iseed
       write(6,*)"First seeds X(k): mrg8dnz2"
       do k=1,8
       x(k)=ishft(xx(k),-1)
       write(6,*)"(",k,")  ",x(k)
       enddo
       iflag=1
      endif
c     main loop starts here.
c     copy seeds to right by kmax (<1024)
      do k=1,8
      x(kmax+k)=x(k)
      enddo
      nn=mod(n,kmax)
      do j=1,n-nn,kmax
      do k=kmax,1,-1
      k1=k+4
      s1=0
      s2=0
      do i=1,4
      s1=s1+a(i)*x(k+i)
      s2=s2+a(i+4)*x(k1+i)
      enddo
      s=iand(s1,mask)+ishft(s1,-31)+iand(s2,mask)+ishft(s2,-31)
      s=iand(s,mask)+ishft(s,-31)
      x(k)=iand(s,mask)+ishft(s,-31)
      ran(j+kmax-k)=dble(x(k))*rnorm
      enddo
c     copy the latest seeds in X to original location
      do k=1,8
      x(kmax+k)=x(k)
      enddo
      enddo
c
       if (nn.gt.0) then
      do i=1,nn
      s1=a(1)*x(1)+a(2)*x(2)+a(3)*x(3)+a(4)*x(4)
      s2=a(5)*x(5)+a(6)*x(6)+a(7)*x(7)+a(8)*x(8)
      s=iand(s1,mask)+ishft(s1,-31)+iand(s2,mask)+ishft(s2,-31)
      s=iand(s,mask)+ishft(s,-31)
      x(8)=x(7)
      x(7)=x(6)
      x(6)=x(5)
      x(5)=x(4)
      x(4)=x(3)
      x(3)=x(2)
      x(2)=x(1)
      x(1)=iand(s,mask)+ishft(s,-31)
      ran(n-nn+i)=dble(x(1))*rnorm
      enddo
      endif
      return
      end
ccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine mcg64ni(iseed,iran,n)
ccccccccccccccccccccccccccccccccccccccccccccccccc
c MCG algorithm from Knuth p.107-108  #26.
Copyright (C) 2006, Kenichi Miura,   All rights reserved. 
c Only to be used for initializing mrg8dn or mrg8dnz2.
c Always start with "iseed".
c  10-6-06   K.Miura
c     real*8 ran(1)
      integer*4 iran(1)
      integer*8 x,ia/6364136223846793005_8/
      integer*4 iseed
      real*8 rnorm/4.656612873077393d-10/
c     write(6,*)"A=",ia
        if(iseed.eq.0)iseed=97531
        x=iseed
      do k=1,n
      x=ia*x
c     ran(k)=dble(ishft(x,-33))*rnorm
      iran(k)=ishft(x,-32)
      enddo
      return
      end
c==========  end of program  =======================


