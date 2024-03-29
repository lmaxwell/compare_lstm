#include "armednn.h"

using namespace Armednn;


void ex1()
{
    auto dense=Registry::get().get_op("Dense").create("dense-0");
    
    std::shared_ptr<Data> d1=std::make_shared<Data>();
    std::shared_ptr<Data> d2=std::make_shared<Data>();

    Matrix temp=Matrix::Identity(8,8);
    d1->allocate(8,8);
    d1->get()=temp;

    DataPtr inputs;
    inputs.push_back(std::move(d1));


    ConfigMap config;
    config.insert({"activation",{(std::string)"tanh"}});
    config.insert({"dim0",{(uint32_t)8}});
    config.insert({"dim1",{(uint32_t)8}});
    ParamMap param;
    param.insert({"weight",{Matrix::Identity(8,8)}});
    param.insert({"bias",{Matrix::Ones(1,8)}});

    Arm arm(config,param);


    std::unique_ptr<Armed> armed(new Armed(arm,std::move(dense)));

    std::unique_ptr<Node> node1(new Node(inputs,std::move(armed)));

    node1->run();
    INFO<<node1->id();
    INFO<<node1->output(0)->get().data();
    INFO<<std::endl<<node1->output(0)->get();

    node1->run();
    INFO<<node1->output(0)->get().data();
    INFO<<std::endl<<node1->output(0)->get();
}



/*
std::unique_ptr<Node> make_node(DataPtr&& inputs, Arm& arm, std::string type, std::string name)
{
    auto op=make_op(
}
*/

void ex2()
{

    auto dense=make_op("Dense","dense-0");

    std::shared_ptr<Data> d1=std::make_shared<Data>();
    std::shared_ptr<Data> d2=std::make_shared<Data>();

    Matrix temp=Matrix::Identity(8,8);
    d1->allocate(8,8);
    d1->get()=temp;

    DataPtr inputs;
    inputs.push_back(std::move(d1));


    ConfigMap config;
    config.insert({"activation",{(std::string)"tanh"}});
    config.insert({"dim0",{(uint32_t)8}});
    config.insert({"dim1",{(uint32_t)8}});
    ParamMap param;
    param.insert({"weight",{Matrix::Identity(8,8)}});
    param.insert({"bias",{Matrix::Ones(1,8)}});

    Arm arm(config,param);


    auto armed=make_armed(arm,std::move(dense));

    auto node1=make_node(inputs,std::move(armed));

    INFO<<node1->id();
    node1->run();
    INFO<<std::endl<<node1->output(0)->get();

}


void ex3()
{

    auto dense=make_op("Dense","dense-0");

    /*
    std::shared_ptr<Data> d1=std::make_shared<Data>();
    std::shared_ptr<Data> d2=std::make_shared<Data>();

    d1->allocate(L,C);
    d1->get()=temp;

    DataPtr inputs;
    inputs.push_back(std::move(d1));
    */

    int L=4;
    int C=12;
    Matrix temp=Matrix::Identity(L,C);

    auto input_node=make_input("input");
    input_node->feed(temp);


    ConfigMap config;
    config.insert({"activation",{(std::string)"tanh"}});
    config.insert({"dim0",{(uint32_t)C}});
    config.insert({"dim1",{(uint32_t)C}});
    ParamMap param;
    param.insert({"weight",{Matrix::Identity(C,C)}});
    param.insert({"bias",{Matrix::Ones(1,C)}});

    Arm arm(config,param);


    auto armed=make_armed(arm,std::move(dense));

    auto node0=make_node(input_node->output(),std::move(armed));

    auto dense_1=make_op("Dense","dense-1");


    armed=make_armed(arm,std::move(dense_1));
    auto node1=make_node(node0->output(),std::move(armed));

    auto split_0=make_op("Split","split-0");
    ConfigMap config_split_0;
    config_split_0.insert({"num_split",{(uint32_t)2}});
    Arm arm_split_0(config_split_0);
    auto armed_split_0=make_armed(arm_split_0,std::move(split_0));
    auto node2=make_node(node1->output(),std::move(armed_split_0));


    auto concat_0=make_op("Concat","concat-0");
    ConfigMap config_concat_0;
    config_concat_0.insert({"num_input",{(uint32_t)2}});
    Arm arm_concat_0(config_concat_0);
    auto armed_concat_0=make_armed(arm_concat_0,std::move(concat_0));
    auto node3=make_node(node2->output(),std::move(armed_concat_0));


    auto fpool_0=make_op("Fpooling","fpool-0");
    ConfigMap config_fpool_0;
    config_fpool_0.insert({"output_channels",{(uint32_t)(C/2)}});
    Arm arm_fpool_0(config_fpool_0);
    auto armed_fpool_0=make_armed(arm_fpool_0,std::move(fpool_0));
    auto init_state_node=make_state("init_state",Matrix::Zero(1,C/2));
    auto node4=make_node({node2->output(0),node2->output(1),init_state_node->output(0)},std::move(armed_fpool_0));



    INFO<<node0->id();
    INFO<<node1->id();
    INFO<<node2->id();
    INFO<<node3->id();
    INFO<<node4->id();

    node0->run();
    node1->run();
    node2->run();
    node3->run();


    //run init_state zero
    init_state_node->run();
    node4->run();

    // state is perserved here
    node4->run();
    node4->run();
    node4->run();
    node4->run();


    // run init_state again to zero
    init_state_node->run();
    node4->run();


    INFO<<std::endl<<node1->output(0)->get();
    INFO<<std::endl<<node2->output(0)->get();
    INFO<<std::endl<<node2->output(1)->get();
    INFO<<std::endl<<node3->output(0)->get();
    INFO<<std::endl<<node4->output(0)->get();

    INFO<<"node0 "<<node0->id();
    for(uint32_t i=0;i<node0->input().size();i++)
        INFO<<node0->input(i)->id();
    INFO<<"node1 "<<node1->id();
    for(uint32_t i=0;i<node1->input().size();i++)
        INFO<<node1->input(i)->id();

    INFO<<"node2 "<<node2->id();
    for(uint32_t i=0;i<node2->input().size();i++)
        INFO<<node2->input(i)->id();
    /*
    INFO<<"node3 "<<node3->id();
    for(uint32_t i=0;i<node3->input().size();i++)
        INFO<<node3->input(i)->id();
    */
}

void ex4()
{


    int L=4;
    int C=12;

    auto input_node=make_input("input");


    ConfigMap config;
    config.insert({"activation",{(std::string)"tanh"}});
    config.insert({"dim0",{(uint32_t)C}});
    config.insert({"dim1",{(uint32_t)C}});
    ParamMap param;
    param.insert({"weight",{Matrix::Identity(C,C)}});
    param.insert({"bias",{Matrix::Ones(1,C)}});

    Arm arm(config,param);

    auto dense_0=make_node("Dense",arm,input_node->output(),"dense-0");
    auto dense_1=make_node("Dense",arm,dense_0->output(),"dense-1");
    auto dense_2=make_node("Dense",arm,dense_1->output(),"dense-2");
    auto dense_3=make_node("Dense",arm,dense_2->output(),"dense-3");


    Matrix temp=Matrix::Identity(L,C);
    input_node->feed(temp);

    dense_0->run();
    dense_1->run();
    dense_2->run();
    dense_3->run();

    INFO<<dense_3->output(0)->get();

}

void ex5()
{


    int L=4;
    int C=12;

    auto input_node=make_input("input");


    ConfigMap config;
    config.insert({"activation",{(std::string)"tanh"}});
    config.insert({"dim0",{(uint32_t)C}});
    config.insert({"dim1",{(uint32_t)C}});
    ParamMap param;
    param.insert({"weight",{Matrix::Identity(C,C)}});
    param.insert({"bias",{Matrix::Ones(1,C)}});

    Arm arm(config,param);

    auto dense_0=make_node("Dense",arm,input_node->output(),"dense-0");
    auto dense_1=make_node("Dense",arm,dense_0->output(),"dense-1");
    auto dense_2=make_node("Dense",arm,dense_1->output(),"dense-2",dense_0->output()); //share output with dense_0
    auto dense_3=make_node("Dense",arm,dense_2->output(),"dense-3",dense_1->output()); //share output with dense_1


    Matrix temp=Matrix::Identity(L,C);
    input_node->feed(temp);

    dense_0->run();
    dense_1->run();
    dense_2->run();
    dense_3->run();

    INFO<<dense_3->output(0)->get();

}

void ex6()
{


    int L=2000;
    int C=256;

    auto input_node=make_input("input");

    uint32_t filter_width=10;
    ConfigMap config_conv1d;
    config_conv1d.insert({"filter_width",{filter_width}});
    config_conv1d.insert({"input_channels",{(uint32_t)C}});
    config_conv1d.insert({"output_channels",{(uint32_t)C}});
    config_conv1d.insert({"padding",{(std::string)"same"}});
    config_conv1d.insert({"activation",{(std::string)"tanh"}});
    ParamMap param_conv1d;
    param_conv1d.insert({"weight",{Matrix::Identity(C*filter_width,C)}});
    param_conv1d.insert({"bias",{Matrix::Ones(1,C)}});
    Arm arm_conv1d(config_conv1d,param_conv1d);
    auto conv1d_0=make_node("Conv1d",arm_conv1d,input_node->output(),"conv1d-0");

    ConfigMap config;
    config.insert({"activation",{(std::string)"tanh"}});
    config.insert({"dim0",{(uint32_t)C}});
    config.insert({"dim1",{(uint32_t)C}});
    ParamMap param;
    param.insert({"weight",{Matrix::Identity(C,C)}});
    param.insert({"bias",{Matrix::Ones(1,C)}});

    Arm arm(config,param);

    auto dense_1=make_node("Dense",arm,conv1d_0->output(),"dense-1");
    auto dense_2=make_node("Dense",arm,dense_1->output(),"dense-2"); 
    auto dense_3=make_node("Dense",arm,dense_2->output(),"dense-3");



    Timer timer;

    for(int i=0;i<6;i++)
    {
        Matrix temp=Matrix::Random(L,C);
        input_node->feed(temp);
        INFO<<"begin";
        timer.reset();
        conv1d_0->run();
        std::cout<<"conv1d end cost:"<< timer.elapsed()<<std::endl;
        timer.reset();
        dense_1->run();
        dense_2->run();
        dense_3->run();
        std::cout<<"all end cost:"<< timer.elapsed()<<std::endl;
        L=L+100;
    }

    //INFO<<dense_3->output(0)->get();

}

void ex7()
{


    int L=1000;
    int C=1024;

    auto input_node=make_input("input");

    ConfigMap config_lstm;
    config_lstm.insert({"input_channels",{(uint32_t)C}});
    config_lstm.insert({"output_channels",{(uint32_t)C}});
    config_lstm.insert({"bidirection",{(bool)true}});

    ParamMap param_lstm;
    param_lstm.insert({"weightR",{Matrix::Identity(C,C*2)}});
    param_lstm.insert({"weightW",{Matrix::Identity(C,C*4)}});
    param_lstm.insert({"bias",{Matrix::Ones(1,C*4)}});
    Arm arm_lstm(config_lstm,param_lstm);

    auto state_0=make_state("state0",Matrix::Zero(1,C));
    state_0->run();
    auto state_1=make_state("state1",Matrix::Zero(1,C));
    state_1->run();
    auto lstm_0=make_node("LSTM",arm_lstm,{input_node->output(0),state_0->output(0),state_1->output(0)},"lstm-0");



    Timer timer;

    for(int i=0;i<6;i++)
    {
        Matrix temp=Matrix::Random(L,C);
        input_node->feed(temp);
        INFO<<"begin";
        timer.reset();
        lstm_0->run();
        std::cout<<"lstm end cost:"<< timer.elapsed()<<std::endl;
        L=L+1;
    }

    //INFO<<dense_3->output(0)->get();


}
/*
REGISTER_OP(LSTM).add_config<uint32_t>("input_channels","input channels")
                .add_config<uint32_t>("output_channels","output channels")
                .add_config<bool>("bidirection","bidirection: true | false")
                .add_param("weightW",lstm_weightW_func) 
                .add_param("weightR",lstm_weightR_func) 
                .add_param("bias",lstm_bias_func)
                .set_num_input("1")
                .set_num_output("1")
                .set_num_state("2");
*/


int main()
{

    /*
    ex1();
    ex2();
    ex3();
    ex4();
    ex5();
    */
    ex6();
    ex7();



    std::unordered_map<std::string,ConfigMap> test;

}
