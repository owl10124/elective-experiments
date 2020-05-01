#include <gtkmm/togglebutton.h>
#include <gtkmm/window.h>

using namespace std;
using namespace Gtk;
using namespace Glib;

class Hello: public Window {
    ToggleButton button = ToggleButton(ustring("试试"));
    void onButtonClicked() {
        button.set_label("hello");
    }
    public:
    Hello(){
        set_title(ustring("Testing"));
        
        button.signal_clicked().connect(sigc::mem_fun(*this, &Hello::onButtonClicked));
        add(button);
        button.show();
    }
    virtual ~Hello(){}
};

int main() {
    auto app = Application::create();

    Hello w;
    w.set_default_size(256,256);

    return app->run(w);
}