// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Functor.h                                                   (C) 2000-2005 */
/*                                                                           */
/* Fonctor.                                                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_FUNCTOR_WITH_ADDRESS_H
#define ARCANE_UTILS_FUNCTOR_WITH_ADDRESS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/IFunctorWithAddress.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief FunctorWithAddress associé à une méthode d'une classe \a T.
 */
template<typename T>
class FunctorWithAddressT
: public IFunctorWithAddress
{
 public:
	
  typedef void (T::*FuncPtr)(); //!< Type du pointeur sur la méthode

 public:
	
  //! Constructeur
  FunctorWithAddressT(T* object,
                      FuncPtr funcptr): m_function(funcptr),
                                        m_object(object){}

  virtual ~FunctorWithAddressT() { }

 protected:

  //! Exécute la méthode associé
   void executeFunctor()
    {
      (m_object->*m_function)();
    }
  
  /*!
   * \internal
   * \brief Retourne l'adresse de la méthode associé.
   * \warning Cette méthode ne doit être appelée que par HYODA
   * et n'est pas valide sur toutes les plate-formes.
   */
  void* functorAddress()
  {
    //GG: NE PAS FAIRE CELA, car trop dependant de la plateforme
#if defined(__x86_64__) && defined(ARCANE_OS_LINUX)
    long unsigned int *func=(long unsigned int*)&m_function;
    //printf("\t\33[7m m_object @%p, m_function @%p=0x%lx 0x%lx (sizeof=%ld)\33[m\n\r", m_object, func, *func, *(func+1), sizeof(m_function));
    // Par exemple pour la fonction d'un module utilisateur (depuis executeFunctor):
    // rcx = 0x79
    // rdx = this
    // rax = *(m_object)
    // rdi = m_function.__delta (=0)
    // movl     (%rax,%rdi,1),%rax => m_object->$vtable (@ offset 0)
    // rax = m_object->$vtable
    // rdi=m_function.__delta+m_object ( = 0+m_object)
    // movl -1(%rcx,%rax,1),%rcx => rcx=0x004280c0!
    long unsigned int pfn=*func;
    long unsigned int of7=(pfn-1)>>3;
    //long unsigned int delta=*(func+1);
    long unsigned int *module_vtable=(long unsigned int*)((long unsigned int*)&(*m_object))[0];
    //printf("\t\33[7mpfn=0x%lx, delta=0x%lx module_vtable @ %p\33[m\n\r", pfn,delta,module_vtable);
    //for(int i=0;i<20;++i) printf("\t\t\33[7vtable[%ld]=0x%lx\33[m\n\r",i,module_vtable[i]);
    // Si le bit de poid faible est à 1, c'est qu'il y a af
    if ((pfn&1)==1){
      //printf("\t\t\33[7mfunctorAddress @ 0x%lx\33[m\n\r",module_vtable[of7]);
      return (void*) module_vtable[of7];
    }
    return (void*) pfn;
#else
    return 0;
#endif
    
  }
  
 public:
  FuncPtr m_function; //!< Pointeur vers la méthode associée.
  T* m_object; //!< Objet associé.
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

