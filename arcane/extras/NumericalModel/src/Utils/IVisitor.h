#ifndef IVISITOR_H
#define IVISITOR_H

//BEGIN_NAMESPACE_PROJECT

/**
 * Interface des visitor
 */
class IVisitor
{
public:
  /** Constructeur de la classe */
  IVisitor() {}
  
  /** Destructeur de la classe */
  virtual ~IVisitor() {}
  
public:
  /** 
   *  Initialise 
   */
  virtual void init() = 0;
  
};


//END_NAMESPACE_PROJECT

#endif
