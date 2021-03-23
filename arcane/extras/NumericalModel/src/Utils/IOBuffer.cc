// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#include "Utils/Utils.h"
#include "Utils/IOBuffer.h"

/*
 ***********************************************************************
 * Class: StringTokenizer                                              *
 * By Arash Partow - 2000                                              *
 * URL: http://www.partow.net/programming/stringtokenizer/index.html   *
 *                                                                     *
 * Copyright Notice:                                                   *
 * Free use of this library is permitted under the guidelines and      *
 * in accordance with the most current version of the Common Public    *
 * License.                                                            *
 * http://www.opensource.org/licenses/cpl.php                          *
 *                                                                     *
 ***********************************************************************
*/

#ifndef INCLUDE_STRINGTOKENIZER_H
#define INCLUDE_STRINGTOKENIZER_H


#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>


class StrTokenizer
{

   public:

    StrTokenizer(const std::string& _str, const std::string& _delim);
   ~StrTokenizer(){};

    int         countTokens();
    bool        hasMoreTokens();
    std::string nextToken();
    int         nextIntToken();
    double      nextFloatToken();
    std::string nextToken(const std::string& delim);
    std::string remainingString();
    std::string filterNextToken(const std::string& filterStr);

   private:

    std::string  token_str;
    std::string  delim;

};

#endif

 
StrTokenizer::StrTokenizer(const std::string& _str, const std::string& _delim)
{

   if ((_str.length() == 0) || (_delim.length() == 0)) return;

   token_str = _str;
   delim     = _delim;

  /*
     Remove sequential delimiter
  */
   std::string::size_type curr_pos = 0;

   while(true)
   {
      if ((curr_pos = token_str.find(delim,curr_pos)) != std::string::npos)
      {
         curr_pos += delim.length();

         while(token_str.find(delim,curr_pos) == curr_pos)
         {
            token_str.erase(curr_pos,delim.length());
         }
      }
      else
       break;
   }

   /*
     Trim leading delimiter
   */
   if (token_str.find(delim,0) == 0)
   {
      token_str.erase(0,delim.length());
   }

   /*
     Trim ending delimiter
   */
   curr_pos = 0;
   if ((curr_pos = token_str.rfind(delim)) != std::string::npos)
   {
      if (curr_pos != (token_str.length() - delim.length())) return;
      token_str.erase(token_str.length() - delim.length(),delim.length());
   }

}


int StrTokenizer::countTokens()
{

   std::string::size_type prev_pos = 0;
   int num_tokens        = 0;

   if (token_str.length() > 0)
   {
      num_tokens = 0;

      std::string::size_type curr_pos = 0;
      while(true)
      {
         if ((curr_pos = token_str.find(delim,curr_pos)) != std::string::npos)
         {
            num_tokens++;
            prev_pos  = curr_pos;
            curr_pos += delim.length();
         }
         else
          break;
      }
      return ++num_tokens;
   }
   else
   {
      return 0;
   }

}


bool StrTokenizer::hasMoreTokens()
{
   return (token_str.length() > 0);
}


std::string StrTokenizer::nextToken()
{

   if (token_str.length() == 0)
     return "";

   std::string  tmp_str = "";
   std::string::size_type pos     = token_str.find(delim,0);

   if (pos != std::string::npos)
   {
      tmp_str   = token_str.substr(0,pos);
      token_str = token_str.substr(pos+delim.length(),token_str.length()-pos);
   }
   else
   {
      tmp_str   = token_str.substr(0,token_str.length());
      token_str = "";
   }

   return tmp_str;
}


int StrTokenizer::nextIntToken()
{
   return atoi(nextToken().c_str());
}


double StrTokenizer::nextFloatToken()
{
   return atof(nextToken().c_str());
}


std::string StrTokenizer::nextToken(const std::string& delimiter)
{
   if (token_str.length() == 0)
     return "";

   std::string  tmp_str = "";
   std::string::size_type pos     = token_str.find(delimiter,0);

   if (pos != std::string::npos)
   {
      tmp_str   = token_str.substr(0,pos);
      token_str = token_str.substr(pos + delimiter.length(),token_str.length() - pos);
   }
   else
   {
      tmp_str   = token_str.substr(0,token_str.length());
      token_str = "";
   }

   return tmp_str;
}


std::string StrTokenizer::remainingString()
{
   return token_str;
}


std::string StrTokenizer::filterNextToken(const std::string& filterStr)
{
   std::string  tmp_str    = nextToken();
   std::string::size_type currentPos = 0;

   while((currentPos = tmp_str.find(filterStr,currentPos)) != std::string::npos)
   {
      tmp_str.erase(currentPos,filterStr.length());
   }

   return tmp_str;
}

/***************************************************************************************/

template<>
void 
_Split::
split(const String& strbuffer,Array<String>& buffer,const String& sep,Integer size)
{
  std::string sbuff(strbuffer.localstr()) ;
  std::string ss(sep.localstr()) ;
  StrTokenizer strtok = StrTokenizer(sbuff,ss);
  if(size==0)
  {
    size = strtok.countTokens() ;
    buffer.resize(size) ;
  }
  Integer i = 0 ;
  while(strtok.hasMoreTokens())
  {
    buffer[i] = String(strtok.nextToken()) ;
    i++ ;
    if(i==size) break ;
  }
}

template<>
void 
_Split::
split(const String& strbuffer,Array<Real>& buffer,const String& sep,Integer size)
{
  std::string sbuff(strbuffer.localstr()) ;
  std::string ss(sep.localstr()) ;
  StrTokenizer strtok = StrTokenizer(sbuff,ss);
  if(size==0)
  {
    size = strtok.countTokens() ;
    buffer.resize(strtok.countTokens()) ;
  }
  Integer i = 0 ;
  while(strtok.hasMoreTokens())
  {
    buffer[i] = strtok.nextFloatToken() ;
    i++ ;
    if(i==size) break ;
  }
}

template<>
void 
_Split::
split(const String& strbuffer,Array<Integer>& buffer,const String& sep,Integer size)
{
  std::string sbuff(strbuffer.localstr()) ;
  std::string ss(sep.localstr()) ;
  StrTokenizer strtok = StrTokenizer(sbuff,ss);
  if(size==0)
  {
    size = strtok.countTokens() ;
    buffer.resize(strtok.countTokens()) ;
  }
  Integer i = 0 ;
  while(strtok.hasMoreTokens())
  {
    buffer[i] = strtok.nextIntToken() ;
    i++ ;
    if(i==size) break ;
  }
}

/***************************************************************************************/

Real IOFile::
getReal()
{
  Real v = 0.;
  (*m_stream) >> ws >> v;
  if (m_stream->good())
    return v;
  throw IOException("IOFile::getReal()","Bad Real");
}

Integer IOFile::
getInteger()
{
  Integer v = 0;
  (*m_stream) >> ws >> v;
  if (m_stream->good())
    return v;
  throw IOException("IOFile::getInteger()","Bad Integer");
}
const char* IOFile::
getNextLine()
{
  while (m_stream->good()){
    m_stream->getline(m_buf,sizeof(m_buf)-1);
    if (m_stream->eof())
      break;
    bool is_comment = true;
    if (m_buf[0]=='\n' || m_buf[0]=='\r')
      continue;
    // Regarde si un caract?re de commentaire est pr?sent
    for( int i=0; i<BUFSIZE && m_buf[i]!='\0'; ++i ){
      if (!isspace(m_buf[i])){
        is_comment = (m_buf[i]=='#');
        break;
      }
    }
    if (!is_comment){
      
      // Supprime le '\n' ou '\r' final
      for( int i=0; i<BUFSIZE && m_buf[i]!='\0'; ++i ){
        //cout << " V=" << m_buf[i] << " I=" << (int)m_buf[i] << "\n";
        if (m_buf[i]=='\n' || m_buf[i]=='\r'){
          m_buf[i] = '\0';
          break ;
        }
      }
      return m_buf ;
    }
  }
  throw IOException("IfpVtkFile::getNexLine()","Unexpected EndOfFile");
}

template<> 
void IOFile::getValue(String& str)
{
  (*m_stream) >> ws >> str;
  if (m_stream->good())
    return ;
  throw IOException("IOFile::getValue()","Bad Type");
}

template<> 
void IOFile::getValue(Real& x) { x= getReal() ; }
template<> 
void IOFile::getValue(Integer& i) { i = getInteger() ; }
 
