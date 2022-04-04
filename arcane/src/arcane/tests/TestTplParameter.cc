// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#if 0
/*---------------------------------------------------------------------------*/
/*                                                                           */
/* Teste l'implémentation des classes templates avec paramètre templates.    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/ModuleInclude.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class T>
class TraitsArrayT
{
 public:
	typedef ArrayT<T> type;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<template<class> class ContainerT>
class ITestWrite
{
 public:
 
	typedef typename ContainerT<Real>::type      ContainerReal;
	typedef typename ContainerT<Real3>::type     ContainerReal3;
	typedef typename ContainerT<Real3x3>::type   ContainerReal3x3;
	typedef typename ContainerT<size_type>::type ContainerSize;
	typedef typename ContainerT<Integer>::type   ContainerInteger;
	typedef typename ContainerT<String>::type    ContainerString;

 private:

	template<class U>
	void _dump_size(const U& v)
		{
			cerr << "Size " << v.size() << '\n';
		}
 public:
	virtual void write(ContainerReal& v)
		{ _dump_size(v); }
	virtual void write(ContainerSize& v)
		{ _dump_size(v); }
	virtual void write(ContainerInteger& v)
		{ _dump_size(v); }
	virtual void write(ContainerReal3& v)
		{ _dump_size(v); }
	virtual void write(ContainerReal3x3& v)
		{ _dump_size(v); }
	virtual void write(ContainerString& v)
		{ _dump_size(v); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern void
_test_tpl_write()
{
	ITestWrite< TraitsArrayT > itw;
	ArrayT<Integer> int_list(5);
	
	itw.write(int_list);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#endif
