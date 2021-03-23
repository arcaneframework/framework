// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef TABLE_H
#define TABLE_H

//BEGIN_NAMESPACE_PROJECT

/**
 * Interface des visitor
 */
class Table
{
public:
  /** Constructeur de la classe */
  Table(String name,int size,Real* x,Real* y) 
  {
    m_name = name ;
    m_size = size ;
    if(size<0)
    {
        m_x = NULL ;
        m_y = NULL ;
        return ;
    }
    else
    {
        m_x = new Real[m_size] ;
        m_y = new Real[m_size] ;
        for(UInt32 i=0;i<m_size;i++)
        {
            m_x[i] = x[i] ;
            m_y[i] = y[i] ;
        }
    }
  }
  
  Table(String name,int size,Real* x,Real* y, bool invx) 
  {
    m_name = name ;
    m_size = size ;
    if(size<0)
    {
        m_x = NULL ;
        m_y = NULL ;
        return ;
    }
    else
    {
        m_x = new Real[m_size] ;
        m_y = new Real[m_size] ;
        if(invx)
        {
            for(UInt32 i=0;i<m_size;i++)
            {
                m_x[i] = x[m_size-1-i] ;
                m_y[i] = y[m_size-1-i] ;
            }
        }
        else
        {
            for(UInt32 i=0;i<m_size;i++)
            {
                m_x[i] = x[i] ;
                m_y[i] = y[i] ;
            }
        }
    }
  }
  
  /** Destructeur de la classe */
  virtual ~Table() 
  {
    if(m_size<0) return ;
    delete [] m_x ;
    delete [] m_y ;
  }

public :
  virtual Table* getInverseFunction(String name)
  {
    //check monotonie
    if(m_size==0) return NULL ;
    Real monotonie = (m_y[1]-m_y[0]) ;
    UInt32 i = 1 ;
    while(i<m_size-2)
    {
        if(monotonie*(m_y[i+1]-m_y[i])<0) return NULL ;
        i++ ;
    }
    Table* invTable = NULL ;
    if(monotonie>0) 
        invTable = new Table(name,m_size,m_y,m_x) ;
    else 
        invTable = new Table(name,m_size,m_y,m_x,true) ;
    return invTable ;
  } 
  
  virtual void invY()
  {
    if(m_size==0) return ;
    for(UInt32 i=0;i<m_size;i++)
        if(m_y[i]!=0.) m_y[i] = 1./m_y[i] ;
  }
  
  virtual void multX(Real factor)
  {
    if(m_size==0) return ;
    for(UInt32 i=0;i<m_size;i++)
        m_x[i] *= factor ;
  }
  
  virtual void multY(Real factor)
  {
    if(m_size==0) return ;
    for(UInt32 i=0;i<m_size;i++)
        m_y[i] *= factor ;
  }
  
public:
    Real eval(const Real x)
    {
        if(m_size==0) return FloatInfo<Real>::maxValue() ;
        if(m_size==1) return m_y[0] ;
        Real y = 0 ;
        Real dy = 0 ;
        if(x<m_x[0])
        {
            evalf(0,x,&y,&dy) ;
        }
        else
        {
            if(x>=m_x[m_size-1])
               evalf(m_size-2,x,&y,&dy) ; 
            else
            {
                int i = 0 ;
                while(x>=m_x[i]) i++ ;
                evalf(i-1,x,&y,&dy) ;
            }
        }
        return y ;
    }
    
    void eval(Real* y,Real* dy, const Real x)
    {
        if(m_size==0) return ;
        if(m_size==1)
        {
            *y = m_y[0] ;
            *dy = 0 ;
            return ;
        }
        if(x<m_x[0])
        {
            evalf(0,x,y,dy) ;
        }
        else
        {
            if(x>=m_x[m_size-1])
               evalf(m_size-2,x,y,dy) ; 
            else
            {
                int i = 0 ;
                while(x>=m_x[i]) i++ ;
                evalf(i-1,x,y,dy) ;
            }
        }
    }
 private :
    inline Real df(int i) { return (m_y[i+1]-m_y[i])/(m_x[i+1]-m_x[i]) ; } 
    inline void evalf(int i, Real x, Real* y, Real* dy ) 
    {
        *dy = df(i) ;
        *y =  m_y[i] + *dy * (x-m_x[i]) ; 
    } 
    
    String m_name ;
    UInt32 m_size ;
    Real* m_x ;
    Real* m_y ;
};

#endif
