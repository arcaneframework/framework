// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IosFile.h             					                             (C) 2000-2021 */
/*                                                                           */
/* Routines des Lecture/Ecriture d'un fichier.                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IOS_FILE_SERVICES_H
#define ARCANE_IOS_FILE_SERVICES_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Routines des Lecture/Ecriture d'un fichier.
 */
class IosFile
{
 public:
	static const int IOS_BFR_SZE = 8192;
 public:
	IosFile(istream* stream) : m_stream(stream) {}
	const char* getNextLine(const char *);
	const char* getNextLine(void);
	Real getReal(void);
	Integer getInteger(void);
	Int64 getInt64(void);
	bool lookForString(const String& str);
	void checkString(const String& current_value,const String& expected_value);
	void checkString(const String& current_value, const String& expected_value1, const String& expected_value2);
	static bool isEqualString(const String& current_value,const String& expected_value);
	bool isEnd(void);
 private:
	istream* m_stream;
	char m_buf[IOS_BFR_SZE];
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
