// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef IOBUFFER_H_
#define IOBUFFER_H_

#include "Utils/Utils.h"

#include <arcane/utils/Iostream.h>
#include <arcane/utils/Buffer.h>
#include <arcane/utils/StdHeader.h>
#include <arcane/utils/HashTableMap.h>
#include <arcane/utils/ValueConvert.h>
#include <arcane/utils/String.h>
#include <arcane/utils/IOException.h>
#include <arcane/utils/Collection.h>
#include <arcane/utils/Enumerator.h>

#include <sstream>

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IOFile
{
public:
  static const int BUFSIZE = 10000;
public:
  IOFile(istream* stream) : m_stream(stream) {}
  virtual ~IOFile() { }
  const char* getNextLine();
  Real getReal();
  Integer getInteger();
  template<class T> void getValue(T& v) ;
  bool isEnd(){ (*m_stream) >> ws; return m_stream->eof(); }
private:
  istream* m_stream;
  char m_buf[BUFSIZE];
};

template<> 
void IOFile::getValue(Real& x) ;
template<> 
void IOFile::getValue(Integer& i) ;
template<> 
void IOFile::getValue(String& str) ;

template<class T> 
void IOFile::getValue(T& v)
{
  (*m_stream) >> ws >> v;
  if (m_stream->good())
    return ;
  throw IOException("IOFile::getValue()","Bad Type");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


class _Split
{
private:
  template<class T> friend bool readBufferFromStringBuffer(const String& strbuffer,
                                                           Array<T> &buffer,
                                                           const String sep,
                                                           Integer size);
  template<class T> friend bool readBufferFromStringBuffer(const String& strbuffer,
                                                           Array<T> &buffer,
                                                           const String sep);
    
  template<class T>
  static void split(const String& str,Array<T>& buffer,const String& sep,Integer size) ;
};

  
template<>
void _Split::split(const String& str,Array<Integer>& buffer,const String& sep,Integer size) ;
  
template<>
void _Split::split(const String& str,Array<Real>& buffer,const String& sep,Integer size) ;
  
template<>
void _Split::split(const String& str,Array<String>& buffer,const String& sep,Integer size) ;


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class T>
bool readBufferFromFile(const String& file_name,
                        Array<T> &buffer,
                        Integer size)
{
  if(buffer.size()<size)
    {
      cout<<"Buffer size is too small"<<endl ;
      return false ;
    }
  ifstream ifile(file_name.localstr());
  if (!ifile){
    cout << "Unable to open in read mode file '" 
         << file_name.localstr() << "'"<<endl;
    return false;
  }
  IOFile file(&ifile);
  Integer count = 0 ;
  
  try 
    { 
      while(!file.isEnd())
        {
          T x ;
          file.getValue(x) ;
          buffer[count] = x ;
          count++ ;
          if(count==size) break ;  
        }
      if(count<size) cout <<"Warning there are less element than : "<<size<<endl ;
      return true ;
    }
  catch(IOException e) 
    {
      cerr<<"Reading file problem :"<<e<<endl ;
      return false ;
    }
}

/*---------------------------------------------------------------------------*/

template<class T>
bool readBufferFromStringBuffer(const String& strbuffer,
                                Array<T> &buffer,
                                const String sep,
                                Integer size)
{
  if(buffer.size()<size)
    {
      cout<<"Buffer size is too small"<<endl ;
      return false ;
    }
  _Split::split(strbuffer,buffer,sep,size) ;
  return true ;
}

/*---------------------------------------------------------------------------*/

template<class T>
bool readBufferFromStringBuffer(const String& strbuffer,
                                Array<T> &buffer,
                                const String sep)
{
  _Split::split(strbuffer,buffer,sep,0) ;
  return true ;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /*IOBUFFER_H_*/
