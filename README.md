# TABA_2023
2023년 단국대학교 TABA 파이썬 기초 강의 자료

- 파이썬기초 실습 colab: https://colab.research.google.com/drive/1rT-sJ--x9qwiqw58qW3JL9uiRyyKje9V?usp=sharing
- 머신러닝 실습 colab: https://colab.research.google.com/drive/12C0wWOIssynrRWtQW-8hQQvLcV-5VZSo?usp=sharing
- 딥러닝 이론 https://aromatic-money-c6a.notion.site/1abdc2da37ee4b74b6a8edaccf197824?v=16e545fcf70c4e818608dda79c901cf7&pvs=4

* view_classify 코드를 추가했습니다.
  ```
  def view_classify(img, ps, version="MNIST"):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    if version == "MNIST":
        ax2.set_yticklabels(np.arange(10))
    elif version == "Fashion":
        ax2.set_yticklabels(['T-shirt/top',
                            'Trouser',
                            'Pullover',
                            'Dress',
                            'Coat',
                            'Sandal',
                            'Shirt',
                            'Sneaker',
                            'Bag',
                            'Ankle Boot'], size='small');
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
  ```
  
