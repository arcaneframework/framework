// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TestTraceMessageListener.h                                  (C) 2000-2011 */
/*                                                                           */
/* Test d'un ITraceMessageListener.                                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANETEST_TESTTRACEMESSAGELISTENER_H
#define ARCANETEST_TESTTRACEMESSAGELISTENER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ITraceMng.h"

#include "arcane/tests/ArcaneTestGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_BEGIN_NAMESPACE

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class TestTraceMessageListener
: public ITraceMessageListener
{
 public:
  TestTraceMessageListener() : m_total(0){ toto = 0; }
 public:
  virtual bool visitMessage(const TraceMessageListenerArgs& args)
  {
    ConstArrayView<char> str(args.buffer());
    if (args.message()->type()!=Trace::Info)
      return false;
    ++toto;
    if (m_total>20000){
      //if(args.message()->level()>0 && (toto%2)==0)
      if((toto%3)==0)
        return true;
      else{
        String toto(str.data());
        toto = String("REDIRECT <<") + toto + ">>";
        args.message()->parent()->writeDirect(args.message(),toto);
      }
    }
    m_total += str.size();
    //std::cout << "RECEIVING : total=" << m_total << " " << str.begin();
    return false;
  }
 private:
  Integer m_total;
  Integer toto;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

