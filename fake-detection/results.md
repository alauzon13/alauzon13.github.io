
In 2019, The RealReal \-- an “authenticated luxury resale” marketplace \-- sold a fake Dior bag to [Forbes contributor](https://www.forbes.com/sites/richardkestenbaum/2019/10/23/if-fake-bags-are-being-sold-on-the-realreal-how-can-the-resale-business-ever-succeed/) Richard Kestenbaum for $3,600 USD. The inauthenticity was determined by external experts who were trained to spot the tells: the herringbone pattern was slightly off; the handles were a hair too short; the embroidery wasn’t as detailed as it should be. 

When contacted about the legitimacy of the bag, The RealReal said they were not able to definitively confirm, stating that authenticating is both “an art and a science.” Further, when asked about the quality of TRR’s internal authentication process, inside sources said that authenticators did “as good a job as any human can.” 

Being both a data scientist and a fashion lover, these comments stuck with me. Kestenbaum also points out that the average customer doesn’t have access to trained authenticity experts. If there is a scientific element to authenticating, if human error is a concern, and if there is an accessibility gap in authentication services, why not turn to machine learning? With the right data and models, an algorithmic approach could make luxury authentication more consistent, scalable, and accessible. 

## The Big Idea 

As a proof-of-concept approach, I wanted to train a binary image classification model to identify real versus fake versions of a single designer bag. I chose to limit the scope to one style because signs might be bag-specific \-- like the embroidery in the fake Dior sold by The RealReal.

## The bag 

When it came to the bag, I knew immediately which one I wanted to investigate: the Hermès Birkin. The Birkin is a universal symbol of wealth and exclusivity, with prices starting at $10,000 USD. The elusivity of Birkins is maintained by the preconditions of purchase: you can’t just walk into a store and buy one. Instead, customers must first build credit with the brand by purchasing other items before they can be considered as potential buyers of a Birkin. They must develop a relationship with a designated sales associate, who plays a critical role in deciding who is able to purchase a bag. The road to a Birkin can be years long, resulting in some unhappy customers [suing](https://www.nytimes.com/2024/03/21/style/birkin-bag-hermes-lawsuit.html) the French luxury brand for violating antitrust law. 

I was particularly interested in the Hermès Himalaya Birkin, a white-and-grey crocodile version of the famous bag. The Himalaya is regarded as one of the most elusive and expensive handbags in the world, with a diamond-encrusted version being sold for over $450,000 USD in 2022 (the most expensive bag sale to date). Even without diamonds, the white and grey crocodile skin bag sells for about $200,000 new. 

![himalaya-birk](assets/himalaya-birk.webp)
*Diamond Himalaya Birkin, [Sotheby’s](https://www.sothebys.com/en/buy/_hermes-himalaya-niloticus-crocodile-birkin-35-palladium-8f53)*

It’s no wonder that, given what a Himalaya communicates about one's time and income, that they are one of the most-duped bags available on the replica market. Just this past December, a replica for sale on [Walmart](https://www.cbc.ca/news/world/walmart-wirkin-bag-1.7423682) went viral and subsequently sold out, and other versions are readily available on sites like DHgate and AliExpress. 

## The data 

### Collection

Before building the model, I needed to collect images online from both real and fake listings of Himalaya Birkins. Because customers need to be recommended by their sales associates to purchase these bags, they aren’t readily available for purchase on the Hermès website. Instead, I turned to resellers or auction sites with authenticity guarantees. Of course, as mentioned at the beginning of this article, there is a chance that good fakes make their way past authenticators, but for the sake of data collection I assumed any bag for sale on these websites ([Madison Avenue Couture](https://madisonavenuecouture.com/products/hermes-birkin-30-himalaya-niloticus-crocodile-palladium-hardware-6?variant=41321718743134), [Sotheby’s](https://www.sothebys.com/en/buy/_hermes-himalaya-niloticus-crocodile-birkin-35-palladium-8f53), [1st Dibs](https://www.1stdibs.com/fashion/handbags-purses-bags/top-handle-bags/hermes-blanc-himalayan-birkin-30cm-diamond-encrusted-18k-wg-hardware/id-v_24286782/), etc.) was authentic. For the replicas, I looked to websites known for selling fakes ([Pochette Bag](https://www.pochettebag.com/products/hermes-birkin-matte-crocodile-himalaya-silver-hardware), [Baginning](https://baginning.com/white-croc-embossement-leather-handbags-gold-metal-lock-satchel-bag.html?gad_source=1&gad_campaignid=20474646201&gbraid=0AAAAAC11NzCu-WXRXgW3FZZDkRv88mvQO&gclid=CjwKCAjwl_XBBhAUEiwAWK2hzrIMRzvIjP7rvDsHo2HMH-8v0FG-_J89YaobZ1lufHpPy2ECXmluZxoCzV8QAvD_BwE)), as well as Hermes-specific replicators ([Hermes Bags Replica](https://www.hermesbagsreplica.com/hermes-birkin-bags-replica/hermes-birkin-30cm-niloticus-crocodile-himalaya), [Be Hermes](https://www.behermesbags.com/hermes-diamond-birkin-25cm-in-himalaya-niloticus-crocodile-skin.html)). I created two folders, *Real* and *Fake*. For all bags, I saved all available product images into their respective folder. I ended up with 55 images of real handbags and 57 images of fake bags. 

### Background Removal + Augmentation

After inspecting the images, I noticed that product photos of fake bags tended to have busier, non-empty backgrounds, whereas authenticated bags tended to be in front of plain, white backgrounds. Since I wanted to limit my analysis to the bags themselves, I opted to remove the backgrounds from all images. 

After inspecting the images, I noticed that product photos of fake bags tended to have busier, non-empty backgrounds, whereas authenticated bags tended to be in front of plain, white backgrounds. Since I wanted to limit my analysis to the bags themselves, I opted to remove the backgrounds from all images. 

<div style="display: flex; gap: 50px; align-items: center;">
   <div style="text-align: center;">
    <img src="assets/real-background.png" width="190" alt="Real bag">
    <p><em> <a href="https://madisonavenuecouture.com/products/hermes-birkin-30-himalaya-niloticus-crocodile-palladium-hardware-6?variant=41321718743134">Madison Avenue Couture</a></em></p>
  </div>
  <div style="text-align: center;">
    <img src="assets/fake-background.png" width="190" alt="Fake bag">
    <p><em><a href="https://www.hermesbagsreplica.com/hermes-birkin-bags-replica/hermes-birkin-30cm-niloticus-crocodile-himalaya">Hermes Bags Replica</a></em></p>
  </div>
</div>

Small datasets like this one are prone to overfitting. A model is overfit when instead of learning population-level characteristics, it simply memorizes the patterns in the provided training set. To boost the amount of data available, I implemented image augmentation -- which is when in addition to the original data, we include copies of images that have been stretched, flipped, rotated, shifted, etc. Augmentation is a common technique to improve model quality when data is limited. 

![aug](assets/augmentation-diag.png)



### The Model 

### The results

