//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;

#if ARCANE_64BIT
using Integer = System.Int64;
#else
using Integer = System.Int32;
#endif
using Real = System.Double;

namespace Arcane
{
  public class Observer : AbstractObserver
  {
    public delegate void ObserverDelegate();
    ObserverDelegate m_functor;
    //IObservable m_observable;

    //~Observer()
    //{
      //Console.WriteLine("OBSERVER DELETE I={0}",m_observable);
      //if (m_observable!=null)
      //base.detach(m_observable);
      //}

    public Observer(ObserverDelegate func)
    {
      m_functor = func;
    }
    public override void ObserverUpdate(IObservable o)
    {
      if (m_functor!=null)
        m_functor();
    }

    //public override void attachToObservable(IObservable obs)
    //{
      //if (m_observable!=null)
        //  throw new ArgumentException("instance is already attached to an observable");
      //base.attachToObservable(obs);
      //m_observable = obs;
      //}
    
    //public override void detach(IObservable obs)
    //{
    //m_observable = null;
    //base.detachFromObservable(obs);
    //}
  }
}
