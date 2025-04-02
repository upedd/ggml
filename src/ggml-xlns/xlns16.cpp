// 16-bit XLNS for C++
// copyright 1999-2025 Mark G. Arnold
// these routines
//    ran on 16-bit Turbo C/C++ (the file may have CR/Lfs from that system)
//    were used in my PhD research and for several later papers on 32-bit LNS
// they were ported to Linux gcc and g++ around 2015 on 32-bit x86
// they were ported again for 64-bit arch in 2025,
//    modified for 16-bit similar to bfloat (see xlns32.cpp for original float-like code)
//    with the xlns16_ideal option
//    with a Mitchell LPVIP method for the non-ideal case
//    with xlns16_alt for streamlined + for modern arch w/ ovfl test
// they are based on similar math foundation (Gaussian logs, sb and db) as Python xlns,
//    but use different internal storage format:
//    +------+-------------------------+
//    | sign | int(log2) . frac(log2)  |
//    +------+-------------------------+
//    the int(log2) is not twos complement; it is offset (logsignmask XORed)
//    for the 16-bit format in this file, this is roughly similar to bfloat16
//    1 sign bit, 8 int(log2) bits, 7 frac(log2) bits
//    there is an exact representation of 0.0, but no subnormals or NaNs

/* PORTABLE CODE STARTS HERE*/

#include <stdio.h>
#include <stdlib.h>
  //typedef unsigned short xlns16;
  //typedef signed short xlns16_signed;
  #ifdef _WIN32
    typedef unsigned __int16 xlns16;
    typedef __int16 xlns16_signed;
  #else
    typedef u_int16_t xlns16;
    typedef int16_t xlns16_signed;
  #endif
  #define xlns16_zero          0x0000
  #define xlns16_scale         0x0080
  #define xlns16_logmask       0x7fff
  #define xlns16_signmask      0x8000
  #define xlns16_logsignmask   0x4000
  #define xlns16_canonmask     0x8000
  #define xlns16_sqrtmask      0x2000
  #define xlns16_esszer        0x0500
  #define xlns16_canonshift    15

#define xlns16_sign(x)  ((x) & xlns16_signmask)
#define xlns16_neg(x)   ((x) ^ xlns16_signmask)
#define xlns16_abs(x)   ((x) & xlns16_logmask)
#define xlns16_recip(x) (xlns16_sign(x)|xlns16_abs((~x)+1))
#define xlns16_sqrt(x)   (xlns16_abs(((xlns16_signed)((x)<<1))/4)^xlns16_sqrtmask)
#define xlns16_canon(x) ((x)^(-((x)>>xlns16_canonshift)|xlns16_signmask))

inline xlns16 xlns16_overflow(xlns16 xlns16_x, xlns16 xlns16_y, xlns16 xlns16_temp)
{       //printf("%d %d %d\n",xlns16_temp,xlns16_x,xlns16_y);
	if (xlns16_logsignmask&xlns16_temp)
	{
		return (xlns16_signmask&(xlns16_x^xlns16_y));
	}
	else
	{
		return (xlns16_signmask&(xlns16_x^xlns16_y))| xlns16_logmask;
	}
}

inline xlns16 xlns16_mul(xlns16 x, xlns16 y)
{
   xlns16 xlns16_temp;
   //xlns16_x = x;
   //xlns16_y=y;
   xlns16_temp=(xlns16_logmask&(x))+(xlns16_logmask&(y))-xlns16_logsignmask;
   return  (xlns16_signmask&(xlns16_temp)) ? xlns16_overflow(x,y,xlns16_temp)
                                       :(xlns16_signmask&(x^y))|xlns16_temp;
}

inline xlns16 xlns16_div(xlns16 x, xlns16 y)
{
   xlns16 xlns16_temp;
   //xlns16_x = x;
   //xlns16_y=y;
   xlns16_temp=(xlns16_logmask&(x))-(xlns16_logmask&(y))+xlns16_logsignmask;
   return  (xlns16_signmask&(xlns16_temp)) ? xlns16_overflow(x,y,xlns16_temp)
                                       :(xlns16_signmask&(x^y))|xlns16_temp;
}

#ifdef xlns16_ideal
  #define xlns16_sb xlns16_sb_ideal
  #define xlns16_db xlns16_db_ideal
  #include <math.h>
  inline xlns16 xlns16_sb_ideal(xlns16_signed z)
  {
	return ((xlns16) ((log(1+ pow(2.0, ((double) z) / xlns16_scale) )/log(2.0))*xlns16_scale+.5));
  }
  inline xlns16 xlns16_db_ideal(xlns16_signed z)
  {
	return ((xlns16_signed) ((log( pow(2.0, ((double) z) / xlns16_scale) - 1 )/log(2.0))*xlns16_scale+.5));
  }
#else
  #define xlns16_sb xlns16_sb_premit
  #define xlns16_db xlns16_db_premit
  #define xlns16_F 7

  #include <math.h>
  inline xlns16 xlns16_db_ideal(xlns16_signed z)  //only for singularity
  {
	return ((xlns16_signed) ((log( pow(2.0, ((double) z) / xlns16_scale) - 1 )/log(2.0))*xlns16_scale+.5));
  }
  inline xlns16 xlns16_mitch(xlns16 z)
  {
     return (((1<<xlns16_F)+(z&((1<<xlns16_F)-1)))>>(-(z>>xlns16_F)));
  }

  inline xlns16 xlns16_sb_premit_neg(xlns16_signed zi)   //was called premitchnpi(zi): assumes zi<=0
  {
    xlns16 postcond;
    xlns16 z;
    postcond = (zi <= -(3<<xlns16_F))? 0: (zi >= -(3<<(xlns16_F-2))? -1: +1);
    z = ((zi<<3) + (zi^0xffff) + 16)>>3;
    return (zi==0)?1<<xlns16_F: xlns16_mitch(z) + postcond;
    //return ((zi==0)?1<<xlns16_F: (((1<<xlns16_F)+(z&((1<<xlns16_F)-1)))>>(-(z>>xlns16_F)))+postcond );
  }

  inline xlns16 xlns16_db_premit_neg(xlns16_signed z)   //assumes zi<0
  {
    xlns16_signed precond;
    precond = (z < -(2<<xlns16_F))?
                    5<<(xlns16_F-3):                //  0.625
                    (z >> 2) + (9 << (xlns16_F-3));//  .25*zr + 9/8
    return (-z >= 1<<xlns16_F)?-xlns16_mitch(z+precond): xlns16_db_ideal(-z)+z; // use ideal for singularity
  }
  inline xlns16 xlns16_sb_premit(xlns16_signed zi)   //assumes zi>=0
  {
    return xlns16_sb_premit_neg(-zi)+zi;
  }
  inline xlns16 xlns16_db_premit(xlns16_signed z)   //assumes zi>0
  {
    return xlns16_db_premit_neg(-z)+z;
  }
#endif


#ifdef xlns16_alt

inline xlns16 xlns16_add(xlns16 x, xlns16 y)
{
    xlns16 minxyl, maxxy, xl, yl, usedb, adjust, adjustez;
    xlns16_signed z;
    xl = x & xlns16_logmask;
    yl = y & xlns16_logmask;
    minxyl = (yl>xl) ? xl : yl;
    maxxy  = (xl>yl) ? x  : y;
    z = minxyl - (maxxy&xlns16_logmask);
    usedb = xlns16_signmask&(x^y);
    #ifdef xlns16_ideal
     float pm1 = usedb ? -1.0 : 1.0;
     adjust = z+((xlns16_signed)(log(pm1+pow(2.0,-((double)z)/xlns16_scale))/log(2.0)*xlns16_scale+.5));
    #else
     //adjust = usedb ? xlns16_db_neg(z) :
     //                 xlns16_sb_neg(z);
     xlns16_signed precond = (usedb==0) ? ((-z)>>3) :          // -.125*z
                (z < -(2<<xlns16_F)) ? 5<<(xlns16_F-3):        //  0.625
                                (z >> 2) + (9 << (xlns16_F-3));//  .25*z + 9/8
     xlns16_signed postcond = (z <= -(3<<xlns16_F)) ? 0:
                            z >= -(3<<(xlns16_F-2)) ? -(1<<(xlns16_F-6)) :
                                                      +(1<<(xlns16_F-6));
     xlns16_signed mitch = (-z >= 1<<xlns16_F)||(usedb==0) ? xlns16_mitch(z+precond) :
                                          -xlns16_db_ideal(-z)-z; // use ideal for singularity
     adjust = usedb ? -mitch : (z==0) ? 1<<xlns16_F : mitch + postcond;
    #endif
    adjustez = (z < -xlns16_esszer) ? 0 : adjust;
    return ((z==0) && usedb) ?
                     xlns16_zero :
                     xlns16_mul(maxxy, xlns16_logsignmask + adjustez);
}

#else

//++++ X-X ERROR fixed

xlns16 xlns16_add(xlns16 x, xlns16 y)
{
	xlns16 t;
	xlns16_signed z;

	z = (x&xlns16_logmask) - (y&xlns16_logmask);
	if (z<0)
	{
		z = -z;
		t = x;
		x = y;
		y = t;
	}
	if (xlns16_signmask&(x^y))
	{
		if (z == 0)
			return xlns16_zero;
		if (z < xlns16_esszer)
			return xlns16_neg(y + xlns16_db(z));
		else
			return xlns16_neg(y+z);
	}
	else
	{
		return y + xlns16_sb(z);
	}
}
#endif

#define xlns16_sub(x,y) xlns16_add(x,xlns16_neg(y))

/*END OF PORTABLE CODE*/

/*START OF PORTABLE CODE THAT DEPENDS ON <math.h>*/

#include <math.h>

xlns16 fp2xlns16(float x)
{
	if (x==0.0)
		return(xlns16_zero);
	else if (x > 0.0)
		return xlns16_abs((xlns16_signed) ((log(x)/log(2.0))*xlns16_scale))
		       ^xlns16_logsignmask;
	else
		return (((xlns16_signed) ((log(fabs(x))/log(2.0))*xlns16_scale))
			  |xlns16_signmask)^xlns16_logsignmask;
}

float xlns162fp(xlns16 x)
{
	if (xlns16_abs(x) == xlns16_zero)
		return (0.0);
	else if (xlns16_sign(x))
		return (float) (-pow(2.0,((double) (((xlns16_signed) (xlns16_abs(x)-xlns16_logsignmask))))
					/((float) xlns16_scale)));
	else {
		return (float) (+pow(2.0,((double) (((xlns16_signed) (xlns16_abs(x)-xlns16_logsignmask))))
					/((float) xlns16_scale)));
	}
}

	//else if (xlns16_sign(x))
		//return (float) (-pow(2.0,((double) (((xlns16_signed) xlns16_abs(x^xlns16_logsignmask))<<1)/2)
		//			/((float) xlns16_scale)));
	//else {
	//	return (float) (+pow(2.0,((double) (((xlns16_signed) xlns16_abs(x^xlns16_logsignmask))<<1)/2)
	//				/((float) xlns16_scale)));

/*END OF PORTABLE CODE THAT DEPENDS ON <math.h>*/




#include <iostream>

class xlns16_float {
    xlns16 x;
 public:
    friend xlns16_float operator+(xlns16_float , xlns16_float );
    friend xlns16_float operator+(float, xlns16_float );
    friend xlns16_float operator+(xlns16_float , float);
    friend xlns16_float operator-(xlns16_float , xlns16_float );
    friend xlns16_float operator-(float, xlns16_float );
    friend xlns16_float operator-(xlns16_float , float);
    friend xlns16_float operator*(xlns16_float , xlns16_float );
    friend xlns16_float operator*(float, xlns16_float );
    friend xlns16_float operator*(xlns16_float , float);
    friend xlns16_float operator/(xlns16_float , xlns16_float );
    friend xlns16_float operator/(float, xlns16_float );
    friend xlns16_float operator/(xlns16_float , float);
    xlns16_float operator=(float);
    friend xlns16 xlns16_internal(xlns16_float );
    friend float xlns16_2float(xlns16_float );
    friend xlns16_float float2xlns16_(float);
    friend std::ostream& operator<<(std::ostream&, xlns16_float );
    friend xlns16_float operator-(xlns16_float);
    friend xlns16_float operator+=(xlns16_float &, xlns16_float);
    friend xlns16_float operator+=(xlns16_float &, float);
    friend xlns16_float operator-=(xlns16_float &, xlns16_float);
    friend xlns16_float operator-=(xlns16_float &, float);
    friend xlns16_float operator*=(xlns16_float &, xlns16_float);
    friend xlns16_float operator*=(xlns16_float &, float);
    friend xlns16_float operator/=(xlns16_float &, xlns16_float);
    friend xlns16_float operator/=(xlns16_float &, float);
    friend xlns16_float sin(xlns16_float);
    friend xlns16_float cos(xlns16_float);
    friend xlns16_float exp(xlns16_float);
    friend xlns16_float log(xlns16_float);
    friend xlns16_float atan(xlns16_float);
    friend xlns16_float abs(xlns16_float);
    friend xlns16_float sqrt(xlns16_float);
//    friend xlns16_float operator-(xlns16_float);
    friend int operator==(xlns16_float arg1, xlns16_float arg2)
      {
       return (arg1.x == arg2.x);
      }
    friend int operator!=(xlns16_float arg1, xlns16_float arg2)
      {
       return (arg1.x != arg2.x);
      }
    friend int operator<=(xlns16_float arg1, xlns16_float arg2)
      {
       return (xlns16_canon(arg1.x)<=xlns16_canon(arg2.x));
      }
    friend int operator>=(xlns16_float arg1, xlns16_float arg2)
      {
       return (xlns16_canon(arg1.x)>=xlns16_canon(arg2.x));
      }
    friend int operator<(xlns16_float arg1, xlns16_float arg2)
      {
       return (xlns16_canon(arg1.x)<xlns16_canon(arg2.x));
      }
    friend int operator>(xlns16_float arg1, xlns16_float arg2)
      {
       return (xlns16_canon(arg1.x)>xlns16_canon(arg2.x));
      }
    friend int operator==(xlns16_float arg1, float arg2);
    friend int operator!=(xlns16_float arg1, float arg2);
    friend int operator<=(xlns16_float arg1, float arg2);
    friend int operator>=(xlns16_float arg1, float arg2);
    friend int operator<(xlns16_float arg1, float arg2);
    friend int operator>(xlns16_float arg1, float arg2);
  };




/*access function for internal representation*/

xlns16 xlns16_internal(xlns16_float y) {
    return y.x;
}


float xlns16_2float(xlns16_float y) {
	return xlns162fp(y.x);
}

#define xlns16_cachesize 1024
xlns16 xlns16_cachecontent[xlns16_cachesize];
float xlns16_cachetag[xlns16_cachesize];
long xlns16_misses=0;
long xlns16_hits=0;
#define xlns16_cacheon 1

xlns16_float float2xlns16_(float y) {
	xlns16_float z;
	unsigned char * fpbyte;
	int addr;
	fpbyte=(unsigned char *)(&y);
	addr = (fpbyte[2])^(fpbyte[3]<<2);
	if ((xlns16_cachetag[addr] ==  y)&&xlns16_cacheon)
	{
//	  printf("hit  %f  %02x %02x %02x %02x addr=%d\n",y, fpbyte[0],fpbyte[1],fpbyte[2],fpbyte[3],addr);
	  z.x = xlns16_cachecontent[addr];
	  xlns16_hits++;
	}
	else
	{
//	  printf("miss %f  %02x %02x %02x %02x addr=%d\n",y, fpbyte[0],fpbyte[1],fpbyte[2],fpbyte[3],addr);
	  z.x = fp2xlns16(y);
	  xlns16_cachecontent[addr] = z.x;
	  xlns16_cachetag[addr] = y;
	  xlns16_misses++;
	}
//	getchar();
	return z;
}


/*overload stream output << operator*/

//#include <ostream>
std::ostream& operator<< (std::ostream& s, xlns16_float  y) {
    return s << xlns16_2float(y);
}

xlns16_float operator-(xlns16_float arg1) {
   xlns16_float z;
   z.x=xlns16_neg(arg1.x);
   return z;
}



xlns16_float operator+(xlns16_float arg1, xlns16_float arg2) {
   xlns16_float z;
   z.x=xlns16_add(arg1.x,arg2.x);
   return z;
}

xlns16_float operator-(xlns16_float arg1, xlns16_float arg2) {
   xlns16_float z;
   z.x=xlns16_sub(arg1.x,arg2.x);
   return z;
}

xlns16_float operator*(xlns16_float arg1, xlns16_float arg2) {
   xlns16_float z;
   z.x=xlns16_mul(arg1.x,arg2.x);
   return z;
}

xlns16_float operator/(xlns16_float arg1, xlns16_float arg2) {
   xlns16_float z;
   z.x=xlns16_div(arg1.x,arg2.x);
   return z;
}


/*operators with auto type conversion*/

xlns16_float operator+(float arg1, xlns16_float arg2) {
   return float2xlns16_(arg1)+arg2;
}

xlns16_float operator+(xlns16_float arg1, float arg2) {
   return arg1+float2xlns16_(arg2);
}


xlns16_float operator-(float arg1, xlns16_float arg2) {
   return float2xlns16_(arg1)-arg2;
}

xlns16_float operator-(xlns16_float arg1, float arg2) {
   return arg1-float2xlns16_(arg2);
}

xlns16_float operator*(float arg1, xlns16_float arg2) {
   return float2xlns16_(arg1)*arg2;
}

xlns16_float operator*(xlns16_float arg1, float arg2) {
   return arg1*float2xlns16_(arg2);
}


xlns16_float operator/(float arg1, xlns16_float arg2) {
   return float2xlns16_(arg1)/arg2;
}

xlns16_float operator/(xlns16_float arg1, float arg2) {
   return arg1/float2xlns16_(arg2);
}

/*comparisons with conversion seems not to inline OK*/

int operator==(xlns16_float arg1, float arg2)
      {
       return arg1 == float2xlns16_(arg2);
      }
int operator!=(xlns16_float arg1, float arg2)
      {
       return arg1 != float2xlns16_(arg2);
      }
int operator<=(xlns16_float arg1, float arg2)
      {
       return arg1<=float2xlns16_(arg2);
      }
int operator>=(xlns16_float arg1, float arg2)
      {
       return arg1>=float2xlns16_(arg2);
      }
int operator<(xlns16_float arg1, float arg2)
      {
       return arg1<float2xlns16_(arg2);
      }
int operator>(xlns16_float arg1, float arg2)
      {
       return arg1>float2xlns16_(arg2);
      }

/*With and without convert:  +=, -=, *=, and /= */

xlns16_float operator+=(xlns16_float & arg1, xlns16_float arg2) {
   arg1 = arg1+arg2;
   return arg1;
}

xlns16_float operator+=(xlns16_float & arg1, float arg2) {
   arg1 = arg1+float2xlns16_(arg2);
   return arg1;
}



xlns16_float operator-=(xlns16_float & arg1, xlns16_float arg2) {
   arg1 = arg1-arg2;
   return arg1;
}

xlns16_float operator-=(xlns16_float & arg1, float arg2) {
   arg1 = arg1-float2xlns16_(arg2);
   return arg1;
}


xlns16_float operator*=(xlns16_float & arg1, xlns16_float arg2) {
   arg1 = arg1*arg2;
   return arg1;
}

xlns16_float operator*=(xlns16_float & arg1, float arg2) {
   arg1 = arg1*float2xlns16_(arg2);
   return arg1;
}


xlns16_float operator/=(xlns16_float & arg1, xlns16_float arg2) {
   arg1 = arg1/arg2;
   return arg1;
}

xlns16_float operator/=(xlns16_float & arg1, float arg2) {
   arg1 = arg1/float2xlns16_(arg2);
   return arg1;
}



/*assignment with type conversion*/


//maybe should use cache here
xlns16_float xlns16_float::operator=(float rvalue) {
//   x = fp2xlns16(rvalue);
     x = float2xlns16_(rvalue).x;
   return *this;
}



// functions computed ideally by convert to/from FP


inline xlns16_float sin(xlns16_float x)
{
	return float2xlns16_(sin(xlns16_2float(x)));
}

inline xlns16_float cos(xlns16_float x)
{
	return float2xlns16_(cos(xlns16_2float(x)));
}

// exp and log can be implemented more efficiently in LNS but
// this is just cookie cutter ideal implementation at present

inline xlns16_float exp(xlns16_float x)
{
	return float2xlns16_(exp(xlns16_2float(x)));
}

inline xlns16_float log(xlns16_float x)
{
	return float2xlns16_(log(xlns16_2float(x)));
}

inline xlns16_float atan(xlns16_float x)
{
	return float2xlns16_(atan(xlns16_2float(x)));
}

// the following have efficient macro implementations

inline xlns16_float sqrt(xlns16_float x)
{
	xlns16_float result;
	result.x = xlns16_sqrt(x.x);
	return result;
}

inline xlns16_float abs(xlns16_float x)
{
	xlns16_float result;
	result.x = xlns16_abs(x.x);
	return result;
}
