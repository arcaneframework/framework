// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CppNameDemangler.cc                                         (C) 2000-2021 */
/*                                                                           */
/* Classe pour 'demangler' un nom correspondant à un type C++.               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/CppNameDemangler.h"

#include <cstdlib>

#ifdef __GNUG__
#define ARCANE_HAS_CXXABI_DEMANGLER
#endif

#ifdef ARCANE_HAS_CXXABI_DEMANGLER
#include <cxxabi.h>
#endif

#include <iostream>

/*!
  la libstdc++ de GCC fournit une méthode abi::_cxa_demangle() pour transformer
  un nom 'manglé' en un nom intelligible.
  
  La page 'https://gcc.gnu.org/onlinedocs/libstdc++/latest-doxygen/a00026.html#a0f77048f40022ee20f49f773defc9c27'
  décrit la documentation de cette fonction.

  \warning Le buffer passé en paramètre de abi::__cxa_demangle() doit avoir été
  alloué par malloc() car il peut être réalloué via realloc() si nécessaire.

  Pour le compilateur Microsoft, std::type_info::name() retourne un nom intelligible
  (i.e. déjà démanglé).
*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class CppNameDemangler::Impl
{
 public:
  Impl(size_t len)
  {
    m_allocated_size = len;
    m_allocated_buffer = (char*)::malloc(m_allocated_size);
  }
  ~Impl()
  {
    if (m_allocated_buffer)
      ::free(m_allocated_buffer);
  }

 public:
  char* m_allocated_buffer = nullptr;
  size_t m_allocated_size = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CppNameDemangler::
CppNameDemangler()
{
  _init(1024);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CppNameDemangler::
CppNameDemangler(size_t buf_len)
{
  _init(buf_len);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CppNameDemangler::
~CppNameDemangler()
{
  delete m_p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CppNameDemangler::
_init(size_t len)
{
  m_p = new Impl(len);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const char* CppNameDemangler::
demangle(const char* mangled_name)
{
  // Values for status:
  //  0 : The demangling operation succeeded.
  // -1 : A memory allocation failure occurred
  // -2 : mangled_name is not a valid name under the C++ ABI mangling rules
  // -3 : One of the arguments is invalid.

#ifdef ARCANE_HAS_CXXABI_DEMANGLER
  int status = 0;
  //std::cerr << "DEMANGLE !! s=" << m_p->m_allocated_size << "\n";
  char* new_buf = abi::__cxa_demangle(mangled_name, m_p->m_allocated_buffer,
                                      &m_p->m_allocated_size, &status);
  //std::cerr << "NEW_SIZE=" << m_p->m_allocated_size << " status=" << status << "\n";
  if (status == 0 && new_buf)
    m_p->m_allocated_buffer = new_buf;
  return new_buf;
#else
  return mangled_name;
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
