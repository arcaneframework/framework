/*
 * main.c
 *
 *  Created on: Nov 25, 2020
 *      Author: gratienj
 */

#define F2C_
#if defined(_F2C) | defined(F2C_)
#define F2C(function) function##_
#else
#define F2C(function) function
#endif

//typedef long int uid_type ;

extern void F2C(test)() ;

int main([[maybe_unused]] int argc,[[maybe_unused]] char** argv)
{
  F2C(test)() ;

  return 0 ;
}
