// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
// Coverity Models For Arcane
// ///////////////////////
// 
// These models suppress a few common false positives, mostly relating to
// asserts, or assert-like functions and macros.
//
// 1. Build the models with cov-make-library (i.e. build this file)
// 2. Use the compiled user model file when running cov-analyze, using
//    the --user-model-file <filename> option.
//

namespace Arcane
{
// You need to get the namespace exactly right, or the model won't match
// and won't be used.
  
// We don't care about the contents of TraceMessage, but we have to 
// define it or we can't declare the member function we really care about.
class TraceMessage {};
  
class TraceAccessor
{
 public:
  // Treat calling this member like an assert, using the Coverity
  // primitive, __coverity_panic__().
  TraceMessage fatal() const {  __coverity_panic__(); }
  TraceMessage pfatal() const { __coverity_panic__();	}
};

// Do exactly the same thing for this class
class ITraceMng
{
 public:
  TraceMessage fatal() { __coverity_panic__(); }
  TraceMessage pfatal() { __coverity_panic__();	}
};

// Pointer p will be free 'later'
// TODO: A corriger proprement plus tard (contourne une fuite mémoire dans la construction des services générés)
class IServiceFactory2 { };
class IServiceFactoryInfo { };
class ServiceInfo {
 public:
  virtual void addFactory(IServiceFactory2 *p) { __coverity_escape__(p); }
  void setFactoryInfo(IServiceFactoryInfo* p) { __coverity_escape__(p); }
};

void _doAssert(const char*,const char*,const char*,size_t)
{
  __coverity_panic__();
}

} // End namespace Arcane
