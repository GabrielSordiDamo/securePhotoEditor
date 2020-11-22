O haarcascade_frontalface_default.xml pode ser encontrado no link:
    https://www.kaggle.com/gpreda/haar-cascades-for-face-detection?select=haarcascade_frontalface_alt_tree.xml

A a data do haarcascade_frontalface_default.xml pode ser encontrado no link:
    https://github.com/opencv/opencv/tree/master/data


Integrante: Gabriel Sordi Damo

Proposta:
    Um mini editor de fotos, mais ou menos seguro. Onde para registrar
    uma foto pelo editor, o usuario deve ser registrado, e se ja registrado,
    entao identificado entre os registros dos usuarios. O usuario deve entao
    colocar as fotos que deseja editar dentro da pasta rawPhotos, e nao deve mudar
    o nome da mesma. A extensao das fotos deve ser .jpg. As fotos dessa pasta
    serao carregadas. E o usuario pode editalas. Enquanto estiver editando as fotos
    o usuario pode fechar a aplicacao mostrando um objeto rosa para a camera ou apertando
    esc. Nesse momento, entao na pasta edithedPhotos o usuario encontrara as fotos editadas.
    Nessa pasta so serao salvas as fotos que de fato passarem por uma modificacao.
    Quando a aplicao for encerrada o log gerado durante o uso sera armazenado no arquivo
        logs.log, aberto em modo 'w'
        os logs sao -> quem logou e quando
                    -> quem deslogou e quando
                    -> quem foi registrado como novo usuario e quando
                    -> quem editou fotos, quantas foram editadas e quando

LibsAplicadas:
    mahotas - para aplicacao de otsu
    numpy - para correto funcinamento do cv2...
    cv2 - para trabalhar cv em cima das imagens...
    os - deletar arquivos, lidar com path....
    re - trabalhar com Regex...
    glob - identificar arquivos que fecham com um padrao
    time - utilizar time.sleep()
    shutil - mover, copiar arquivos...
    face_recognitios - inteligencia artificial em reconhecimento de faces...
    loggin - trabalhar com logging
    Stack - trabalhar com pilha (implementada na aula de estrutura de dados)

Etapas de processamento:
    O fluxo princiap do programa esta disponivel no link:
        https://lucid.app/lucidchart/invitations/accept/f87510cb-458a-43f0-ae8b-ff5cb18e428d