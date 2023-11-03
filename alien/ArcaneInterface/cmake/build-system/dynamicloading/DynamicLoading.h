#include "gmodule.h"

#ifdef WIN32
#include <ciso646>
#endif

#include <list>
#include <memory>
#include <iostream>
#include <string>
#include <stdexcept>

namespace DynamicLoading
{
   class Loader
   {
   public:

      Loader() {}

       Loader(const Loader&) = delete;
        Loader(Loader&&) = delete;
        void operator=(const Loader&) = delete;
        void operator=(Loader&&) = delete;

        ~Loader()
        {
            for (auto* module : m_loaded_modules) 
            {
                 if (not module) continue;
                 if (not g_module_close(module))
                      std::cout << "WARNING: can not unload module\n";
            }
        }

        void load(std::string name)
        {
            if (not _load(name) && not _load(name + ".dll"))
            {
                std::cout << "ERROR: can not load module '" << name << "'\n";
                throw std::runtime_error("can not load module");
            }
        }

    private:

        bool _load(std::string name)
        {
            auto* path = g_module_build_path(".", name.c_str());
//            std::cout << "** Load Dynamic Library '" << path << "'...";
            auto* gmodule = g_module_open(path, GModuleFlags());
            g_free(path);
            if (not gmodule){
                std::cout << " NOT FOUND\n";
                return false;
            }
            else {
                std::cout << " OK\n";
                m_loaded_modules.push_back(gmodule);
                return true;
            }
        }

    private:

        std::list<GModule*> m_loaded_modules;
    };

    std::shared_ptr<Loader> newLoader()
    {
        return std::make_shared<Loader>();
    }

}
