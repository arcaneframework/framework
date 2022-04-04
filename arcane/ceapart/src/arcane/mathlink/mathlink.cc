// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* mathlink.cc                                                      (C) 2013 */
/*                                                                           */
/*---------------------------------------------------------------------------*/
#include "arcane/IApplication.h"
#include "arcane/IParallelMng.h"

#include "arcane/AbstractService.h"
#include "arcane/FactoryService.h"

#include "arcane/IVariableMng.h"
#include "arcane/SharedVariable.h"
#include "arcane/CommonVariables.h"

#include "arcane/IMesh.h"
#include "arcane/IItemFamily.h"

#include <arcane/MathUtils.h>
#include <arcane/Timer.h>
#include <arcane/IParallelMng.h>
#include <arcane/ITimeLoopMng.h>

#include <mathlink.h>
#include <arcane/mathlink/mathlink.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

// ****************************************************************************
// * mathlink
// ****************************************************************************
mathlink::mathlink(const ServiceBuildInfo & sbi):
  AbstractService(sbi),
  m_sub_domain(sbi.subDomain()),
  mathenv(NULL),
  mathlnk(NULL),
  mathtmr(new Timer(m_sub_domain,"mathlink",Timer::TimerReal)){}


// ****************************************************************************
// * ~mathlink
// ****************************************************************************
mathlink::~mathlink(){}


// ****************************************************************************
// * link
// ****************************************************************************
void mathlink::link(){
  int error;

  // Il n'y a que le master en 0 qui se link à Mathematica
  if (m_sub_domain->parallelMng()->commRank()!=0) return;
  

  debug()<<"[mathlink::link]"<<" linking?"
        <<" (commSize=" << m_sub_domain->parallelMng()->commSize()
        <<", commRank=" << m_sub_domain->parallelMng()->commRank()<<")";
  
  { // Initializes the MathLink environment object and passes parameters in p
    mathenv = MLInitialize((char *)NULL);
    if(mathenv == (MLENV)NULL)
      throw Arcane::FatalErrorException(A_FUNCINFO, "Unable to initialize the MathLink environment");
    debug()<<"[mathlink::link] initialized!";
  }
  
  { // Opens a MathLink connection taking parameters from a character string
    // -linkhost localhost /cea/produits1/mathematica-8.0.0/Executables/math
    // pkill -9 MathKernel
    // SharedMemory
    char master_string[]="-linkname /cea/produits1/mathematica-8.0.0/Executables/math -mathlink -linkmode launch -linkprotocol SharedMemory";
    char slave_string[] ="-linkname /cea/produits1/mathematica-8.0.0/Executables/math -mathlink -linkmode connect -linkprotocol tcpip";
    if (m_sub_domain->parallelMng()->commRank()==0){
      mathlnk = MLOpenString(mathenv, master_string, &error);
    }else{
      mathlnk = MLOpenString(mathenv, slave_string, &error);
    }
    if ((mathlnk==(MLINK)NULL)||(error!=MLEOK))
      throw Arcane::FatalErrorException(A_FUNCINFO, "Unable to create the link");
  }
  
  { // Activates a MathLink connection, waiting for the program at the other end to respond. 
    if (!MLActivate(mathlnk))
      throw Arcane::FatalErrorException(A_FUNCINFO, "Unable to establish communication");
    info()<<"Mathematica launched on rank #"<<m_sub_domain->parallelMng()->commRank();
  }
}


// ****************************************************************************
// * unlink
// ****************************************************************************
void mathlink::unlink(){ 
  info()<<__FUNCTION__<<" MLClose";
  if (!mathlnk) return;
  MLPutFunction(mathlnk, "Exit", 0);
  // Closes a MathLink connection
  MLClose(mathlnk);
  info()<<__FUNCTION__<<" MLDeinitialize";
  if (!mathenv) return;
  // Destructs the MathLink environment object
  MLDeinitialize(mathenv);
}

// ****************************************************************************
// * statics to read outputs
// ****************************************************************************
static int read_and_print_expression( MLINK lp);
static int   read_and_print_atom( MLINK lp, int tag){
	const char *s;
	if( tag == MLTKSTR) putchar( '"');
	if( MLGetString( lp, &s)){
		printf( "%s", s);
		//MLDisownString( lp, s);
    ARCANE_FATAL("MLDisownString is unknown");
	}
	if( tag == MLTKSTR) putchar( '"');
	putchar( ' ');
	return MLError( lp) == MLEOK;
}
static int read_and_print_function( MLINK lp){
	int  len, i;
	static int indent;
	if( ! MLGetArgCount( lp, &len)) return 0;
	indent += 3;
	printf( "\n%*.*s", indent, indent, "");
	if( read_and_print_expression( lp) == 0) return 0;
	printf( "[");
	for( i = 1; i <= len; ++i) {
		if( read_and_print_expression( lp) == 0) return 0;
		if( i < len) printf( ", ");
	}
	printf( "]");
	indent -= 3;
	return 1;
}
static int read_and_print_expression( MLINK lp){
	int tag;
	switch (tag = MLGetNext( lp)) {
	case MLTKSYM:
	case MLTKSTR:
	case MLTKINT:
	case MLTKREAL:
		return read_and_print_atom(lp, tag);
	case MLTKFUNC:
		return read_and_print_function(lp);
	case MLTKERROR:{
     printf("MLTKERROR!\n");
     break;
   }
	default: printf("\nread_and_print_expression default!");
	}
   return 0;
}


// ****************************************************************************
// * skip any packets before the first ReturnPacket 
// ****************************************************************************
void mathlink::skipAnyPacketsBeforeTheFirstReturnPacket(){
  int pkt;
  while( (pkt = MLNextPacket(mathlnk), pkt) && pkt != RETURNPKT) {
    MLNewPacket(mathlnk);
    if (MLError(mathlnk)) mathlink::error();
  }
}


// ****************************************************************************
// * Prime
// * gives the nth prime number
// ****************************************************************************
Integer mathlink::Prime(Integer n){
  mlint64 prime;
  if (n==0) return 1;
  debug()<<"[mathlink::Prime] n=" << n;
  MLPutFunction(mathlnk, "EvaluatePacket", 1L);
  MLPutFunction(mathlnk, "Prime", 1L);
  MLPutInteger64(mathlnk, n);
  MLEndPacket(mathlnk);
  skipAnyPacketsBeforeTheFirstReturnPacket();
  if (MLGetNext(mathlnk)!=MLTKINT) mathlink::error();
  MLGetInteger64(mathlnk, &prime);
  debug()<<"[mathlink::Prime] returning " << prime;
  return prime;
}


// ****************************************************************************
// * tests
// ****************************************************************************
void mathlink::tests(){
  testFactorInteger(7420738134810L);
  UniqueArray<Integer> coefs;
  coefs.add(3);
  coefs.add(4);
  coefs.add(7);
  testLinearProgramming(coefs.view());
}


// ****************************************************************************
// * testFactorInteger
// ****************************************************************************
void mathlink::testFactorInteger(Int64 n){
  info()<<__FUNCTION__<<" is now factoring "<<n;
  {
    int pkt, expt;
    mlint64 prime;
    long len, lenp, k;
    Timer::Sentry ts(mathtmr);
    
    MLPutFunction(mathlnk, "EvaluatePacket", 1L);
    MLPutFunction(mathlnk, "FactorInteger", 1L);
    MLPutInteger64(mathlnk, n);
    MLEndPacket(mathlnk);
    while( (pkt = MLNextPacket(mathlnk), pkt) && pkt != RETURNPKT) {
      MLNewPacket(mathlnk);
      if (MLError(mathlnk)) mathlink::error();
    }
    if (!MLCheckFunction(mathlnk, "List", &len)) mathlink::error();
    for (k = 1; k <= len; k++) {
      if (MLCheckFunction(mathlnk, "List", &lenp)
          &&  lenp == 2
          &&  MLGetInteger64(mathlnk, &prime)
          &&  MLGetInteger(mathlnk, &expt)
          ){
        info()<<prime<<"^"<< expt;
      }else mathlink::error();
    }
  }
  info()<<__FUNCTION__<<" "<<mathtmr->lastActivationTime()<<"s";
}



// ****************************************************************************
// * testLinearProgramming
// ****************************************************************************
void mathlink::testLinearProgramming(ArrayView<Integer> coefs){
  {
    int pkt,sol;
    long len,k;
    Timer::Sentry ts(mathtmr);
    MLPutFunction(mathlnk, "EvaluatePacket", 1);
    MLPutFunction(mathlnk, "LinearProgramming", 5);
    int c[]={1,1,1,1,1,1};
    int m[5][6]={{7, 4, 3, 0, 0, 0},
                 {0, 0, 0, 7, 4, 3},
                 {1, 0, 0, 1, 0, 0},
                 {0, 1, 0, 0, 1, 0},
                 {0, 0, 1, 0, 0, 1}};
    int b[5][2]={{9, -1}, {9, -1}, {1, 0}, {1, 0}, {1, 0}};
    int l[6][2]={{0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}};
    MLPutIntegerList(mathlnk, c, 6);
    MLPutFunction(mathlnk, "List", 5);
    MLPutIntegerList(mathlnk, m[0], 6);
    MLPutIntegerList(mathlnk, m[1], 6);
    MLPutIntegerList(mathlnk, m[2], 6);
    MLPutIntegerList(mathlnk, m[3], 6);
    MLPutIntegerList(mathlnk, m[4], 6);
    MLPutFunction(mathlnk, "List", 5);
    MLPutIntegerList(mathlnk, b[0], 2);
    MLPutIntegerList(mathlnk, b[1], 2);
    MLPutIntegerList(mathlnk, b[2], 2);
    MLPutIntegerList(mathlnk, b[3], 2);
    MLPutIntegerList(mathlnk, b[4], 2);
    MLPutFunction(mathlnk, "List", 6);
    MLPutIntegerList(mathlnk, l[0], 2);
    MLPutIntegerList(mathlnk, l[1], 2);
    MLPutIntegerList(mathlnk, l[2], 2);
    MLPutIntegerList(mathlnk, l[3], 2);
    MLPutIntegerList(mathlnk, l[4], 2);
    MLPutIntegerList(mathlnk, l[5], 2);
    MLPutFunction(mathlnk, "List", 6);
    MLPutSymbol(mathlnk, "Integers");
    MLPutSymbol(mathlnk, "Integers");
    MLPutSymbol(mathlnk, "Integers");
    MLPutSymbol(mathlnk, "Integers");
    MLPutSymbol(mathlnk, "Integers");
    MLPutSymbol(mathlnk, "Integers");
    MLEndPacket(mathlnk);
    while( (pkt = MLNextPacket(mathlnk), pkt) && pkt != RETURNPKT) {
      MLNewPacket(mathlnk);
      if (MLError(mathlnk)) mathlink::error();
    }
    if (!MLCheckFunction(mathlnk, "List", &len)) mathlink::error();
    for (k = 1; k <= len; k++) {
      if (MLGetInteger(mathlnk, &sol)){
        info()<<sol;
      }else mathlink::error();
    }
  }
  info()<<__FUNCTION__<<" "<<mathtmr->lastActivationTime()<<"s";
}


// ****************************************************************************
// * error
// ****************************************************************************
void mathlink::error(){
  if (MLError(mathlnk))
    throw Arcane::FatalErrorException(A_FUNCINFO, MLErrorMessage(mathlnk));
  else
    throw Arcane::FatalErrorException(A_FUNCINFO,"Error detected by mathlink.\n");
}


ARCANE_REGISTER_SUB_DOMAIN_FACTORY(mathlink, mathlink, mathlink);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
