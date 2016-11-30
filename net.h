#ifndef NET_H
#define NET_H


class Net
{
public:
    Net();

    // adding layer to the network
    void add();

    // usual API
    void forward();
    void backward();



};

#endif // NET_H
